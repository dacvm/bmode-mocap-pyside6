
"""Live-preview widget that streams from the selected USB camera into the Qt UI."""

# Standard library helpers for launching a Qt application and accepting CLI args.
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
# Optional typing helper preserved for future extensions or annotations.
from typing import Optional

# Third-party libs for video capture, image encoding, and Qt widgets/layouts.
import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPixmap, QScreen
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox, QWidget

# Generated UI classes that wrap the Qt Designer forms into Python-friendly fields.
from ui.bmode_ui import Ui_Form as Ui_BModeV2
# Class to parse the Calibration XML.
from helpers.xml_stream_parser import (
    CoordinateDefinitionsExtractor,
    VideoDeviceExtractor,
    XmlStreamParser,
)

IMAGE_RECORD_QUEUE_MAXSIZE = 300
JPEG_QUALITY = 85
IMAGE_RECORD_DROP_WINDOW_SECONDS = 2.0
IMAGE_RECORD_DROP_RATE_THRESHOLD = 0.05


# Summary:
# - Build a monotonic timestamp in milliseconds for frame packets.
# - What it does: Uses `time.monotonic()` so image timestamps share the same stable
#   clock source as mocap timestamps for software coupling.
# - Input: None.
# - Returns: Monotonic milliseconds (int).
def _now_ms() -> int:
    # Use a monotonic clock so timestamp deltas are stable if system time changes.
    return int(time.monotonic() * 1000)


# Summary:
# - Lightweight container for one streamed frame and its metadata.
# - What it does: Stores timestamp, dimensions, format label, and raw bytes for UI rendering/recording.
@dataclass
class FramePacket:
    timestamp_ms: int
    width: int
    height: int
    fmt: str
    data: bytes


# Summary:
# - Qt signal bridge used to safely move frames/state from worker loops into the Qt UI thread.
# - What it does: Exposes signals for frame data, running state, and error messages.
class BModeStreamProxy(QObject):
    frame_ready = Signal(object)
    state_changed = Signal(bool, str)
    error_message = Signal(str)
    record_message = Signal(str)
    record_stop = Signal(str)

    # Summary:
    # - Initialize the proxy and assign a stable objectName for slot naming consistency.
    # - Input: `self`.
    # - Returns: None.
    def __init__(self) -> None:
        super().__init__()
        # Give the proxy a name so slot handlers can follow the naming convention.
        self.setObjectName("bmodeStreamProxy")


# Summary:
# - Background worker that owns the camera capture loop.
# - What it does: Opens the camera on a thread, reads frames, and emits FramePacket via the proxy.
class CameraStreamWorker:
    # Summary:
    # - Initialize the worker and its thread state.
    # - Input: `self`, `proxy` (BModeStreamProxy).
    # - Returns: None.
    def __init__(self, proxy: BModeStreamProxy) -> None:
        # Store the proxy so background threads can emit UI-safe signals.
        self._proxy = proxy

        # Track thread state so start/stop are repeat-safe.
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._lock = threading.Lock()
        self._cap: Optional[cv2.VideoCapture] = None
        self._active = False

    # Summary:
    # - Start the camera acquisition thread.
    # - Input: `self`, `camera_index` (int), `fps` (int; used as ms interval).
    # - Returns: None.
    def start(self, camera_index: int, fps: int = 33) -> None:
        # Guard against double-starts to keep the worker repeat-safe.
        with self._lock:
            if self._active:
                return
            self._active = True

            # Create a fresh stop event for this streaming session.
            self._stop_event = threading.Event()

            # Start the background thread so camera I/O never blocks the UI thread.
            self._thread = threading.Thread(
                target=self._run,
                args=(int(camera_index), int(fps), self._stop_event),
                daemon=True,
            )
            # Thread start: kick off the worker loop.
            self._thread.start()

    # Summary:
    # - Stop the active camera thread without blocking the UI for long.
    # - Input: `self`.
    # - Returns: None.
    def stop(self) -> None:
        # Lock to avoid racing against start.
        with self._lock:
            if not self._active:
                return
            self._active = False

            stop_event = self._stop_event
            worker_thread = self._thread

            # Clear references so a new start can rebuild cleanly.
            self._stop_event = None
            self._thread = None

        # Signal the worker loop to exit.
        if stop_event is not None:
            stop_event.set()

        # Join briefly so stop does not freeze the UI.
        if worker_thread is not None:
            worker_thread.join(timeout=0.5)

    # Summary:
    # - Emit a proxy signal safely when the UI may have already been destroyed.
    # - Input: `self`, `emit_fn` (callable), `args` (tuple[object, ...]).
    # - Returns: None.
    def _safe_emit(self, emit_fn, *args) -> None:
        # Avoid crashing if Qt deletes the proxy while the thread is still running.
        try:
            emit_fn(*args)
        except RuntimeError:
            # Ignore emits after shutdown is already in progress.
            return

    # Summary:
    # - Worker loop that opens the camera and streams frames until stopped.
    # - Input: `self`, `camera_index` (int), `fps` (int), `stop_event` (threading.Event).
    # - Returns: None.
    def _run(self, camera_index: int, fps: int, stop_event: threading.Event) -> None:
        # Open the camera inside the worker thread to keep the UI responsive.
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self._cap = cap

        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            self._cap = None
            # Emit the error on the proxy so the UI thread can show the warning.
            self._safe_emit(
                self._proxy.error_message.emit,
                "Failed to open the selected camera. Please try another port.",
            )
            # Emit the stopped state so the UI resets cleanly.
            self._safe_emit(self._proxy.state_changed.emit, False, "camera failed")
            # Clear the active flag so another start can be attempted.
            with self._lock:
                self._active = False
            return

        # Notify the UI that camera streaming is active.
        self._safe_emit(self._proxy.state_changed.emit, True, "camera streaming")

        # Convert the requested interval (ms) into seconds for time.sleep.
        frame_interval = max(int(fps), 1) / 1000.0

        try:
            # Keep reading frames until the stop event is set.
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    # Notify the UI once, then break to stop cleanly.
                    self._safe_emit(
                        self._proxy.error_message.emit,
                        "The camera stopped sending frames. The stream has been closed.",
                    )
                    break

                # Convert camera frames to single-channel uint8 because reconstruction expects grayscale.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                height, width = gray.shape

                # Package the grayscale frame so the UI thread can render safely.
                packet = FramePacket(
                    _now_ms(),
                    width,
                    height,
                    "gray8",
                    gray.tobytes(),
                )
                # Emit the frame via the proxy so the UI thread can pick it up.
                self._safe_emit(self._proxy.frame_ready.emit, packet)

                # Throttle the loop to the requested interval.
                time.sleep(frame_interval)
        finally:
            # Always release the camera handle when the loop ends.
            cap.release()
            self._cap = None
            # Notify the UI that camera streaming ended.
            self._safe_emit(self._proxy.state_changed.emit, False, "camera stopped")
            # Clear the active flag so another start can be attempted.
            with self._lock:
                self._active = False


# Summary:
# - UI-thread worker that captures a screen region using QScreen.grabWindow.
# - What it does: Uses a QTimer on the UI thread because grabWindow must run on the GUI thread and is
#   lightweight enough for this use case, so a separate thread is unnecessary.
class ScreenGrabWorker(QObject):
    # Summary:
    # - Initialize the worker with a proxy and a UI-thread timer.
    # - Input: `self`, `proxy` (BModeStreamProxy), `parent` (QObject | None).
    # - Returns: None.
    def __init__(self, proxy: BModeStreamProxy, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        # Store the proxy so we can emit UI-safe signals.
        self._proxy = proxy

        # Create the UI-thread timer that drives screen grabs.
        self._timer = QTimer(self)
        self._timer.setObjectName("bmodeScreenTimer")
        # Signal connection: grab a frame on each timer tick.
        self._timer.timeout.connect(self._on_bmodeScreenTimer_timeout)

        # Cache the current screen and capture rectangle for the active session.
        self._screen: Optional[QScreen] = None
        self._rect: Optional[tuple[int, int, int, int]] = None

    # Summary:
    # - Start grabbing the screen region on a timer.
    # - Input: `self`, `screen` (QScreen), `rect` (tuple[int, int, int, int]), `fps` (int; ms interval).
    # - Returns: None.
    def start(self, screen: QScreen, rect: tuple[int, int, int, int], fps: int = 33) -> None:
        # Avoid double-starts so repeated calls are safe.
        if self._timer.isActive():
            return

        # Validate the input screen before starting the timer.
        if screen is None:
            self._proxy.error_message.emit(
                "No screen was detected. Cannot start screen streaming."
            )
            self._proxy.state_changed.emit(False, "screen failed")
            return

        origin_x, origin_y, size_width, size_height = rect
        if size_width <= 0 or size_height <= 0:
            self._proxy.error_message.emit(
                "Invalid clip rectangle size detected. The stream has been closed."
            )
            self._proxy.state_changed.emit(False, "screen failed")
            return

        # Store the capture settings so the timer callback can use them.
        self._screen = screen
        self._rect = rect

        # Start the UI-thread timer at the requested interval.
        self._timer.setInterval(max(int(fps), 1))
        self._timer.start()

        # Notify the UI that screen streaming is active.
        self._proxy.state_changed.emit(True, "screen streaming")

    # Summary:
    # - Stop the active screen-grab timer and clear cached state.
    # - Input: `self`.
    # - Returns: None.
    def stop(self) -> None:
        # Skip work if the timer is already stopped.
        if not self._timer.isActive():
            return

        # Stop the UI-thread timer so no more grabs occur.
        self._timer.stop()

        # Clear cached screen info so stale settings are not reused.
        self._screen = None
        self._rect = None

        # Notify the UI that screen streaming ended.
        self._proxy.state_changed.emit(False, "screen stopped")

    # Summary:
    # - Slot function that captures a screen frame on each timer tick.
    # - Input: `self`.
    # - Returns: None.
    def _on_bmodeScreenTimer_timeout(self) -> None:
        # Skip if the screen or rect is missing (should not happen while active).
        if self._screen is None or self._rect is None:
            return

        # Ensure the selected screen is still connected.
        if self._screen not in QGuiApplication.screens():
            self._proxy.error_message.emit(
                "The screen is no longer available. The stream has been closed."
            )
            self.stop()
            return

        origin_x, origin_y, size_width, size_height = self._rect
        if size_width <= 0 or size_height <= 0:
            self._proxy.error_message.emit(
                "Invalid clip rectangle size detected. The stream has been closed."
            )
            self.stop()
            return

        # Grab the calibrated region from the screen on the UI thread.
        pixmap = self._screen.grabWindow(
            0, origin_x, origin_y, size_width, size_height
        )
        if pixmap.isNull():
            self._proxy.error_message.emit(
                "Failed to capture the screen region. The stream has been closed."
            )
            self.stop()
            return

        # Convert the pixmap into grayscale so downstream payloads are single-channel uint8.
        qimg = pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
        bytes_per_line = qimg.bytesPerLine()
        expected_bytes_per_line = qimg.width()
        height = qimg.height()

        # Extract the raw bytes from the QImage memory.
        bits = qimg.bits()
        buffer_size = qimg.sizeInBytes()
        if hasattr(bits, "setsize"):
            # PySide builds that expose sip.voidptr need setsize to view the buffer.
            bits.setsize(buffer_size)
            raw = bytes(bits)
        else:
            # Some builds return a memoryview; slice the reported size.
            raw = bytes(bits[:buffer_size])

        if bytes_per_line != expected_bytes_per_line:
            # Remove per-line padding so the payload is tightly packed grayscale bytes.
            packed_lines = []
            for row in range(height):
                start = row * bytes_per_line
                packed_lines.append(raw[start : start + expected_bytes_per_line])
            data = b"".join(packed_lines)
        else:
            data = raw

        # Emit the frame packet for UI rendering.
        packet = FramePacket(
            _now_ms(),
            qimg.width(),
            qimg.height(),
            "gray8",
            data,
        )
        self._proxy.frame_ready.emit(packet)


# Summary:
# - Background worker that writes incoming B-mode frames as JPEG files on a thread.
# - What it does: Accepts frames on the UI thread, enqueues them, and encodes/writes on a worker thread.
class ImageRecordWorker:
    # Summary:
    # - Initialize the image recording worker and its thread state.
    # - Input: `self`, `proxy` (BModeStreamProxy).
    # - Returns: None.
    def __init__(self, proxy: BModeStreamProxy) -> None:
        # Store the proxy so background threads can send UI-safe signals.
        self._proxy = proxy

        # Track whether recording is active so we can ignore frames when stopped.
        self._active = False
        # Store thread resources that are created on start and released on stop.
        self._queue: Optional[queue.Queue] = None
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._session_dir: Optional[str] = None

        # Track indexing so filenames are sorted by capture order.
        self._frame_index = 0

        # Track drop statistics to detect sustained overload on the UI thread.
        self._record_drop_window_start: Optional[float] = None
        self._record_drop_window_total = 0
        self._record_drop_window_dropped = 0
        self._record_overload_triggered = False

        # Guard start/stop so we do not race if the UI toggles quickly.
        self._lock = threading.Lock()

    # Summary:
    # - Build a unique session directory path for a new recording session.
    # - Input: `record_dir` (str).
    # - Returns: Full session directory path (str).
    @staticmethod
    def _build_session_dir(record_dir: str) -> str:
        # Use UTC in the folder name so sessions are sortable and unambiguous.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        base_name = f"bmode_{timestamp}"
        session_dir = os.path.join(record_dir, base_name)

        # Add a numeric suffix when a folder with the same name already exists.
        if not os.path.exists(session_dir):
            return session_dir

        for suffix in range(1, 1000):
            candidate = os.path.join(record_dir, f"{base_name}_{suffix:03d}")
            if not os.path.exists(candidate):
                return candidate

        # Fall back to the base name if we somehow exhausted all suffixes.
        return session_dir

    # Summary:
    # - Start the JPEG writer thread and reset recording state for a new session.
    # - Input: `self`, `record_dir` (str), `queue_maxsize` (int | None).
    # - Returns: Created session directory path (str).
    def start(
        self,
        record_dir: str,
        queue_maxsize: Optional[int] = None,
    ) -> str:
        # Ensure only one recording session is active at a time.
        with self._lock:
            if self._active:
                return self._session_dir or record_dir
            self._active = True

            # Normalize and ensure the base record directory exists.
            record_dir = os.path.abspath(record_dir)
            try:
                os.makedirs(record_dir, exist_ok=True)
                session_dir = self._build_session_dir(record_dir)
                os.makedirs(session_dir, exist_ok=True)
            except OSError:
                # Reset active state when the directory cannot be created.
                self._active = False
                self._session_dir = None
                raise

            # Reset per-session state before the writer thread starts.
            self._session_dir = session_dir
            self._frame_index = 0
            self._record_drop_window_start = None
            self._record_drop_window_total = 0
            self._record_drop_window_dropped = 0
            self._record_overload_triggered = False

            # Use the requested queue size so external sync can avoid drops.
            if queue_maxsize is None:
                queue_maxsize = IMAGE_RECORD_QUEUE_MAXSIZE
            # Create fresh queue/stop event per session so writer state is isolated.
            self._queue = queue.Queue(maxsize=max(int(queue_maxsize), 0))
            self._stop_event = threading.Event()

            # Start the writer thread so JPEG encoding and disk I/O never block the UI.
            self._thread = threading.Thread(
                target=self._writer_loop,
                args=(session_dir, self._queue, self._stop_event),
                daemon=True,
            )
            # Thread start: begin draining frames to disk asynchronously.
            self._thread.start()

            return session_dir

    # Summary:
    # - Stop the active recording session and release writer resources safely.
    # - Input: `self`, `reason` (str).
    # - Returns: None.
    def stop(self, reason: str = "") -> None:
        # The UI owns user-facing messaging, so we only stop the worker here.
        _ = reason

        # Ensure stop is idempotent and thread-safe.
        with self._lock:
            if not self._active:
                return
            self._active = False

            stop_event = self._stop_event
            writer_thread = self._thread

            # Clear references so new sessions start cleanly.
            self._queue = None
            self._thread = None
            self._stop_event = None
            self._session_dir = None

        # Signal the writer thread to drain and exit.
        if stop_event is not None:
            stop_event.set()

        # Join briefly so we do not freeze the UI while final frames flush.
        if writer_thread is not None:
            writer_thread.join(timeout=0.5)

    # Summary:
    # - Check whether recording is currently active.
    # - Input: `self`.
    # - Returns: True when recording is active (bool).
    def is_active(self) -> bool:
        # Lock to avoid racing a start/stop toggle.
        with self._lock:
            return self._active

    # Summary:
    # - Emit a proxy signal safely when the UI may have already been destroyed.
    # - Input: `self`, `emit_fn` (callable), `args` (tuple[object, ...]).
    # - Returns: None.
    def _safe_emit(self, emit_fn, *args) -> None:
        # Avoid crashing if Qt deletes the proxy while background threads still run.
        try:
            emit_fn(*args)
        except RuntimeError:
            # Ignore emits after the proxy is gone; shutdown is in progress.
            return

    # Summary:
    # - Enqueue a frame for JPEG recording without blocking the UI thread.
    # - Input: `self`, `packet` (FramePacket).
    # - Returns: None.
    def handle_frame(self, packet: FramePacket) -> None:
        # Snapshot state under lock so stop/start does not race frame handling.
        with self._lock:
            if not self._active:
                return
            record_queue = self._queue
            stop_event = self._stop_event

        # Skip enqueue work when the queue is missing or a stop is pending.
        if record_queue is None or (stop_event is not None and stop_event.is_set()):
            return

        # Only pass the minimal payload needed for the writer thread.
        payload = (packet.timestamp_ms, packet.width, packet.height, packet.data)

        # Enqueue without blocking to keep the UI thread responsive.
        dropped = False
        try:
            record_queue.put_nowait(payload)
        except queue.Full:
            dropped = True

        # Track drops over a time window to detect sustained overload.
        now = time.monotonic()
        if self._record_drop_window_start is None:
            self._record_drop_window_start = now
            self._record_drop_window_total = 0
            self._record_drop_window_dropped = 0
        self._record_drop_window_total += 1
        if dropped:
            self._record_drop_window_dropped += 1

        window_elapsed = now - self._record_drop_window_start
        if window_elapsed >= IMAGE_RECORD_DROP_WINDOW_SECONDS:
            drop_rate = 0.0
            if self._record_drop_window_total > 0:
                drop_rate = (
                    self._record_drop_window_dropped / self._record_drop_window_total
                )
            # Reset the window counters for the next measurement period.
            self._record_drop_window_start = now
            self._record_drop_window_total = 0
            self._record_drop_window_dropped = 0

            if drop_rate >= IMAGE_RECORD_DROP_RATE_THRESHOLD:
                if not self._record_overload_triggered:
                    # Stop recording once when sustained overload is detected.
                    self._record_overload_triggered = True
                    self._safe_emit(
                        self._proxy.record_stop.emit,
                        "Recording stopped because the encoder queue is overloaded.",
                    )

    # Summary:
    # - Writer loop that encodes queued frames and saves them as JPEG files.
    # - Input: `self`, `session_dir` (str), `record_queue` (queue.Queue),
    #   `stop_event` (threading.Event).
    # - Returns: None.
    def _writer_loop(
        self,
        session_dir: str,
        record_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        # Drain the queue until stopped, then exit once it is empty.
        while True:
            if stop_event.is_set():
                # If a stop is requested, drain any remaining frames without blocking.
                try:
                    payload = record_queue.get_nowait()
                except queue.Empty:
                    break
            else:
                # When running, wait briefly for the next frame to reduce CPU usage.
                try:
                    payload = record_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

            timestamp_ms, width, height, data = payload
            # Validate against a single-channel payload size because streaming now uses uint8 grayscale.
            expected_size = int(width) * int(height)
            if len(data) < expected_size:
                # Stop recording if frame data is incomplete to avoid corrupt files.
                self._safe_emit(
                    self._proxy.record_stop.emit,
                    "Recording stopped because a frame was incomplete.",
                )
                stop_event.set()
                break

            try:
                # Decode the packed frame buffer as a 2D grayscale image.
                gray = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
            except ValueError:
                # Stop recording if the byte buffer cannot be reshaped.
                self._safe_emit(
                    self._proxy.record_stop.emit,
                    "Recording stopped because a frame could not be decoded.",
                )
                stop_event.set()
                break

            # Encode grayscale JPEG directly to keep disk writes lightweight.
            ok, buffer = cv2.imencode(
                ".jpg",
                gray,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            if not ok:
                self._safe_emit(
                    self._proxy.record_stop.emit,
                    "Recording stopped because JPEG encoding failed.",
                )
                stop_event.set()
                break

            # Build the output path with a stable filename pattern.
            filename = f"frame_{self._frame_index:06d}_{timestamp_ms}.jpg"
            output_path = os.path.join(session_dir, filename)

            try:
                with open(output_path, "wb") as file_handle:
                    file_handle.write(buffer.tobytes())
            except OSError:
                self._safe_emit(
                    self._proxy.record_stop.emit,
                    "Recording stopped because a file could not be written.",
                )
                stop_event.set()
                break

            # Increment the frame index so filenames remain ordered.
            self._frame_index += 1


# Summary: Main Qt window for the B-mode streaming UI.
# What it does: Builds the widgets from the Designer `.ui`, wires signals, and manages stream state.
# Input: Created with no args; uses `self.ui` widgets and internal state to react to user actions.
# Returns: A QWidget subclass instance that can be shown in a QApplication.
class BModeWidget(QWidget):
    """Lightweight window that simply renders the V2 form."""
    # Signal: emit one image packet per received frame for downstream coupling.
    sig_image_packet = Signal(int, object)

    # Summary: Initialize the window and connect UI signals.
    # What it does: Sets up the generated UI, prepares stream workers, and hooks buttons/combos to handlers.
    # Input: `self` (the new window instance).
    # Returns: `None` (constructor).
    def __init__(self) -> None:
        super().__init__()  # Call the QWidget constructor to initialize Qt internals.
        self.ui = Ui_BModeV2()  # Instantiate the generated UI helper for the V2 layout.
        self.ui.setupUi(self)  # Wire the widgets from the .ui file to this QWidget instance.

        # Cache the calibration-derived screen rectangle used for screen streaming.
        self._screen_rect: Optional[tuple[int, int, int, int]] = None

        # Create the proxy and workers that handle acquisition outside the UI widgets.
        self._proxy = BModeStreamProxy()
        self._camera_worker = CameraStreamWorker(self._proxy)
        self._screen_worker = ScreenGrabWorker(self._proxy, parent=self)
        # Create the recording worker so the UI can offload JPEG writes.
        self._image_record_worker = ImageRecordWorker(self._proxy)

        # Cache the most recent frame packet so the UI can render on a timer.
        self._latest_frame_packet: Optional[FramePacket] = None
        # Cache the latest image packet fields for coupled stream consumers.
        self._latest_image_ts_ms: Optional[int] = None
        self._latest_image_data: Optional[object] = None
        # Track the current stream state to keep the open/stop button repeat-safe.
        self._is_streaming = False
        # Track external recording mode so image saving can be driven by mocap timestamps.
        self._external_recording_active = False

        # Timer to throttle rendering so the UI thread stays responsive.
        self._display_timer = QTimer(self)
        self._display_timer.setObjectName("bmodeDisplayTimer")
        self._display_timer.setInterval(33)  # ~33 FPS
        # Signal connection: render the latest cached frame on each tick.
        self._display_timer.timeout.connect(self._on_bmodeDisplayTimer_timeout)

        # Signal connection: cache incoming frames from workers.
        self._proxy.frame_ready.connect(self._on_bmodeStreamProxy_frame_ready)
        # Signal connection: react to start/stop state changes from workers.
        self._proxy.state_changed.connect(self._on_bmodeStreamProxy_state_changed)
        # Signal connection: surface worker errors on the UI thread.
        self._proxy.error_message.connect(self._on_bmodeStreamProxy_error_message)
        # Signal connection: stop recording when the writer thread reports a failure.
        self._proxy.record_stop.connect(self._on_bmodeStreamProxy_record_stop)
        # Signal connection: surface non-fatal recording messages on the UI thread.
        self._proxy.record_message.connect(self._on_bmodeStreamProxy_record_message)

        # Fill the stream port dropdown with any live camera indices we can open.
        self._populate_cameras()

        # Signal connection: update UI when the stream option changes.
        self.ui.comboBox_bmode_streamOption.currentTextChanged.connect(
            self._on_comboBox_bmode_streamOption_changed
        )
        # Apply the current selection immediately so the GUI is consistent on startup.
        self._on_comboBox_bmode_streamOption_changed(
            self.ui.comboBox_bmode_streamOption.currentText()
        )

        # Signal connection: trigger the stream-opening flow on button click.
        self.ui.pushButton_bmode_openStream.clicked.connect(
            self._on_pushButton_bmode_openStream_clicked
        )
        # Signal connection: open a file dialog when the user browses for calibration.
        self.ui.pushButton_bmode_calibBrowse.clicked.connect(
            self._on_pushButton_bmode_calibBrowse_clicked
        )
        # Signal connection: clear the calibration path when the clear button is clicked.
        self.ui.pushButton_bmode_calibClear.clicked.connect(
            self._on_pushButton_bmode_calibClear_clicked
        )
        # Signal connection: clear the record directory when the clear button is clicked.
        self.ui.pushButton_bmode_recorddirClear.clicked.connect(
            self._on_pushButton_bmode_recorddirClear_clicked
        )
        # Signal connection: open a directory picker for the record directory.
        self.ui.pushButton_bmode_recorddirBrowse.clicked.connect(
            self._on_pushButton_bmode_recorddirBrowse_clicked
        )
        # Signal connection: toggle recording when the record button is clicked.
        self.ui.pushButton_bmode_recordStream.clicked.connect(
            self._on_pushButton_bmode_recordStream_clicked
        )

    # Summary: Scan for available cameras and show them in the dropdown.
    # What it does: Tries a small set of camera indices and adds the ones that open successfully.
    # Input: `self`.
    # Returns: `None`.
    def _populate_cameras(self) -> None:
        """List available camera indices in the stream port combo box."""
        # Start with an empty list to avoid stale entries between refreshes.
        self.ui.comboBox_bmode_streamPort.clear()
        for index in range(5):
            # Try to open each index and only keep it if the camera responds.
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if (cap is None) or (not cap.isOpened()):
                if cap is not None:
                    cap.release()
                # Skip indices that either failed to initialize or do not exist.
                continue
            # Add the working camera to the dropdown so the user can select it.
            self.ui.comboBox_bmode_streamPort.addItem(f"Camera {index}", index)
            cap.release()  # Close the camera as we only need to query its availability.

    # Summary: Slot function that updates UI when the stream option changes.
    # What it does: Enables/disables the camera port selector depending on whether we stream a camera or the screen.
    # Input: `self`, `text` (the current stream option label from the combo box).
    # Returns: `None`.
    def _on_comboBox_bmode_streamOption_changed(self, text: str) -> None:
        """Disable the port selector when the user picks the Other screen option."""
        # When streaming to the PC screen, we do not need a camera index, so gray out the selector.
        self.ui.comboBox_bmode_streamPort.setEnabled(text != "Stream Screen (This PC)")

    # Summary: Slot function that opens/closes streaming when the user clicks the button.
    # What it does: Validates the current UI selections, then starts streaming or stops it if already running.
    # Input: `self`.
    # Returns: `None`.
    def _on_pushButton_bmode_openStream_clicked(self) -> None:
        # If we are already streaming, stop immediately so the button is repeat-safe.
        if self._is_streaming:
            self._stop_stream()
            return

        # If the user selected the "Stream Screen" option, ensure the calibration path is set.
        stream_option = self.ui.comboBox_bmode_streamOption.currentText()
        calib_path = self.ui.lineEdit_bmode_calibPath.text().strip()
        if stream_option != "Stream Image" and not calib_path:
            QMessageBox.warning(
                self,
                "Calibration Path Missing",
                "Calibration path is required before streaming the screen.",  # replace this text as needed
            )
            return

        # Ensure we have at least one camera entry to use when streaming the camera.
        if stream_option == "Stream Image" and self.ui.comboBox_bmode_streamPort.count() == 0:
            QMessageBox.warning(
                self,
                "No Camera Found",
                "No available camera ports were detected. Please plug in a camera and try again.",
            )
            return

        # Start the stream after the warning so the flow continues.
        self._start_stream()

    # Summary: Slot function that lets the user choose a calibration XML file.
    # What it does: Opens a file picker, stores the chosen path in the UI, and parses the XML for clip rectangle data.
    # Input: `self`.
    # Returns: `None`.
    def _on_pushButton_bmode_calibBrowse_clicked(self) -> None:
        """Prompt the user to select a calibration XML file and store the path."""
        # Resolve the directory where this Python file lives (i.e., the app folder).
        # Using an absolute path avoids issues when the app is launched from a different working directory.
        base_dir = Path(__file__).resolve().parent
        # Build the path to the local "configs" folder where calibration XMLs are stored.
        configs_dir = base_dir / "configs"
        # Prefer a specific default calibration file so the file dialog opens directly on it.
        # This helps users find the correct XML quickly.
        default_xml = (
            configs_dir
            / "PlusDeviceSet_fCal_Epiphan_NDIPolaris_RadboudUMC_20241219_150400.xml"
        )
        # Choose what initial path the file dialog should show:
        # - If the default XML exists, open the dialog with that file selected.
        # - Otherwise, fall back to the configs directory (so the user can still browse).
        dialog_path = str(default_xml if default_xml.exists() else configs_dir)

        # Show a standard "Open File" dialog.
        # Returns a tuple: (selected_file_path, selected_filter).
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            # Title text shown at the top of the dialog window.
            "Select Calibration XML",
            # Initial directory or filename the dialog opens to.
            dialog_path,
            # Filter list the user can pick from; by default we focus on .xml files.
            "XML Files (*.xml);;All Files (*)",
        )
        # If the user cancels the dialog, Qt returns an empty string; do nothing in that case.
        if not file_path:
            return
        # Store the chosen file path in the UI so other actions (e.g., starting a stream)
        # can read and use this calibration file.
        self.ui.lineEdit_bmode_calibPath.setText(file_path)

        # Parse the XML file to extract the clip rectangle parameters.
        parser = XmlStreamParser(str(file_path))
        parser.register(VideoDeviceExtractor())
        data = parser.parse()

        # Get the ClipRectangleOrigin and ClipRectangleSize.
        video_device = data.get("VideoDevice") or {}
        video_device_attrib = video_device.get("attrib") or {}
        clip_rectangle_origin = video_device_attrib.get("ClipRectangleOrigin")
        clip_rectangle_size = video_device_attrib.get("ClipRectangleSize")
        if not clip_rectangle_origin or not clip_rectangle_size:
            QMessageBox.warning(
                self,
                "Calibration Error",
                "Clip rectangle origin/size is missing in the selected calibration file.",
            )
            self._screen_rect = None
            return
        # Convert the string coordinates into integers for capture.
        origin_x, origin_y = map(int, clip_rectangle_origin.split())
        size_width, size_height = map(int, clip_rectangle_size.split())
        # Cache the rectangle so screen streaming can reuse it.
        self._screen_rect = (origin_x, origin_y, size_width, size_height)

    # Summary: Slot function that clears the calibration path and related cached screen settings.
    # What it does: Removes the calibration file path from the UI and clears any data that depends on it.
    # Input: `self`.
    # Returns: `None`.
    def _on_pushButton_bmode_calibClear_clicked(self) -> None:
        """Clear the calibration path text and reset calibration-derived state."""
        # Clear the stored path so no old file path is left in the UI.
        self.ui.lineEdit_bmode_calibPath.setText("")
        # Reset the placeholder to the default drive hint for the user.
        self.ui.lineEdit_bmode_calibPath.setPlaceholderText("D:\\")
        # Remove any cached calibration data so streaming can't reuse a stale file.
        self._screen_rect = None

    # Summary: Slot function that clears the record directory line edit.
    # What it does: Removes the record directory path so the user can pick a new one.
    # Input: `self`.
    # Returns: `None`.
    def _on_pushButton_bmode_recorddirClear_clicked(self) -> None:
        # UI state change: clear the selected record directory text.
        self.ui.lineEdit_bmode_recorddir.setText("")

    # Summary: Slot function that opens a directory picker for the record directory.
    # What it does: Opens a directory-only dialog and stores the chosen path in the UI.
    # Input: `self`.
    # Returns: `None`.
    def _on_pushButton_bmode_recorddirBrowse_clicked(self) -> None:
        # Use the current text as the starting folder when it exists.
        current_path = self.ui.lineEdit_bmode_recorddir.text().strip()
        # Open a directory-only picker so the user can select a folder.
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Record Directory",
            current_path,
            QFileDialog.ShowDirsOnly,
        )
        # Do nothing if the user cancels the dialog.
        if not selected_dir:
            return
        # Normalize to an absolute path so downstream code is consistent.
        absolute_path = os.path.abspath(selected_dir)
        # UI state change: store the selected directory in the line edit.
        self.ui.lineEdit_bmode_recorddir.setText(absolute_path)

    # Summary: Slot function that starts or stops image recording.
    # What it does: Toggles the JPEG writer thread based on current recording state.
    # Input: `self`.
    # Returns: `None`.
    def _on_pushButton_bmode_recordStream_clicked(self) -> None:
        # Toggle based on current recording state to keep the button behavior predictable.
        if self._image_record_worker.is_active():
            self._stop_recording()
            return
        self._start_recording()

    # Summary: Start a new JPEG recording session on the UI thread.
    # What it does: Validates the record directory, starts the writer thread, and locks UI controls.
    # Input: `self`.
    # Returns: `None`.
    def _start_recording(self) -> None:
        # Require an active stream so recordings stay aligned to live frames.
        if not self._is_streaming:
            QMessageBox.warning(
                self,
                "Stream Not Active",
                "Start streaming before recording.",
            )
            return

        # Validate the record directory so we fail fast on missing paths.
        record_dir = self.ui.lineEdit_bmode_recorddir.text().strip()
        if not record_dir:
            QMessageBox.warning(
                self,
                "Missing Record Directory",
                "Please choose a record directory before recording.",
            )
            return

        # Normalize to an absolute path so sessions are created consistently.
        record_dir = os.path.abspath(record_dir)

        # Ensure the directory exists so the writer can create a session folder.
        try:
            os.makedirs(record_dir, exist_ok=True)
        except OSError:
            QMessageBox.warning(
                self,
                "Record Directory Error",
                "The selected record directory could not be created.",
            )
            return

        if not os.path.isdir(record_dir):
            QMessageBox.warning(
                self,
                "Invalid Record Directory",
                "The selected record directory does not exist.",
            )
            return
        if not os.access(record_dir, os.W_OK):
            QMessageBox.warning(
                self,
                "Record Directory Not Writable",
                "The selected directory is not writable.",
            )
            return

        # Start the writer thread so JPEG encoding and disk I/O stay off the UI thread.
        try:
            self._image_record_worker.start(record_dir)
        except OSError:
            QMessageBox.warning(
                self,
                "Record Directory Error",
                "Failed to create the recording session directory.",
            )
            return

        # UI state change: lock the record directory controls while recording is active.
        self.ui.lineEdit_bmode_recorddir.setEnabled(False)
        self.ui.pushButton_bmode_recorddirBrowse.setEnabled(False)
        self.ui.pushButton_bmode_recorddirClear.setEnabled(False)
        # UI state change: update the record button to show stop intent.
        self.ui.pushButton_bmode_recordStream.setText("Stop Recording")

    # Summary: Stop the active JPEG recording session and release writer resources.
    # What it does: Stops the background thread, restores UI controls, and reports any reason.
    # Input: `self`, `reason` (str).
    # Returns: `None`.
    def _stop_recording(self, reason: str = "") -> None:
        # Avoid redundant work when recording is already stopped.
        if not self._image_record_worker.is_active():
            return

        # Stop the recording worker so it can flush and close cleanly.
        self._image_record_worker.stop(reason)
        # Reset external state so normal per-frame recording can resume later.
        self._external_recording_active = False

        # UI state change: unlock the record directory controls.
        self.ui.lineEdit_bmode_recorddir.setEnabled(True)
        self.ui.pushButton_bmode_recorddirBrowse.setEnabled(True)
        self.ui.pushButton_bmode_recorddirClear.setEnabled(True)
        # UI state change: restore the record button text.
        self.ui.pushButton_bmode_recordStream.setText("Record")

        # Show the stop reason so the user understands why recording ended.
        if reason:
            QMessageBox.warning(self, "Recording Stopped", reason)

    # Summary:
    # - Start an externally-controlled image recording session.
    # - What it does: Starts the image writer thread and marks external mode so snapshots can be driven
    #   by mocap timestamps instead of every incoming frame.
    # - Input: `self`, `record_dir` (str).
    # - Returns: Created session directory path (str).
    def start_external_image_record(self, record_dir: str) -> str:
        # Start the writer thread with an unbounded queue so snapshots are never dropped.
        session_dir = self._image_record_worker.start(record_dir, queue_maxsize=0)
        # Track external state so per-frame auto recording is disabled.
        self._external_recording_active = True
        return session_dir

    # Summary:
    # - Stop an externally-controlled image recording session.
    # - What it does: Clears external mode and stops the image writer thread.
    # - Input: `self`, `reason` (str).
    # - Returns: None.
    def stop_external_image_record(self, reason: str = "") -> None:
        # Clear external mode so per-frame recording logic can resume when needed.
        self._external_recording_active = False
        # Stop the writer thread so file handles flush and close.
        self._image_record_worker.stop(reason)

    # Summary:
    # - Record a snapshot of the latest frame using an injected timestamp.
    # - What it does: Reuses the most recent frame bytes but overrides the timestamp to align
    #   filenames with mocap CSV row times.
    # - Input: `self`, `ts_ms` (int).
    # - Returns: None.
    def record_latest_frame_with_ts(self, ts_ms: int) -> None:
        # Ignore snapshot requests when external recording is not active.
        if not self._external_recording_active:
            return
        # Skip recording when no frame has been cached yet.
        if self._latest_frame_packet is None:
            return

        # Build a new packet with the injected timestamp but the same frame payload.
        latest_packet = self._latest_frame_packet
        snapshot_packet = FramePacket(
            int(ts_ms),
            latest_packet.width,
            latest_packet.height,
            latest_packet.fmt,
            latest_packet.data,
        )
        # Enqueue the snapshot without blocking the UI thread.
        self._image_record_worker.handle_frame(snapshot_packet)

    # Summary: Start streaming based on the selected stream option.
    # What it does: Chooses camera streaming or screen streaming, then starts the appropriate worker.
    # Input: `self`.
    # Returns: `None`.
    def _start_stream(self) -> None:
        """Start the selected stream source via the worker architecture."""
        stream_option = self.ui.comboBox_bmode_streamOption.currentText()
        if stream_option == "Stream Image":
            self._start_camera_stream()
        else:
            self._start_screen_stream()

    # Summary: Start streaming frames from the selected camera.
    # What it does: Resolves the camera index and asks the camera worker to start streaming.
    # Input: `self`.
    # Returns: `None`.
    def _start_camera_stream(self) -> None:
        """Begin camera streaming via the background worker thread."""
        # Read the selected camera index (stored as item data).
        selected_index = self.ui.comboBox_bmode_streamPort.currentData()
        if selected_index is None:
            selected_index = self.ui.comboBox_bmode_streamPort.currentIndex()

        # Start the camera worker; it will emit state changes and frames via the proxy.
        self._camera_worker.start(int(selected_index), fps=33)

    # Summary: Start streaming a calibrated region from the local screen.
    # What it does: Requires calibration rectangle data, asks the user which screen to use, then starts the worker.
    # Input: `self`.
    # Returns: `None`.
    def _start_screen_stream(self) -> None:
        """Begin streaming a cropped region from the local screen."""
        if not self._screen_rect:
            QMessageBox.warning(
                self,
                "Calibration Required",
                "Please load a calibration file before streaming the screen.",
            )
            return

        screen = self._select_screen_for_stream()
        if screen is None:
            if not QGuiApplication.screens():
                QMessageBox.warning(
                    self,
                    "Screen Error",
                    "No screen was detected. Cannot start screen streaming.",
                )
            return

        # Start the screen worker; it runs on the UI thread via QTimer.
        self._screen_worker.start(screen, self._screen_rect, fps=33)

    # Summary: Ask the user which monitor/screen to capture.
    # What it does: Builds a small dropdown list of available screens and returns the chosen QScreen.
    # Input: `self`.
    # Returns: A `QScreen` if the user picks one; otherwise `None`.
    def _select_screen_for_stream(self) -> Optional[QScreen]:

        # Ask Qt for the list of screens currently connected to the machine.
        # Qt will return one QScreen object per monitor.
        screens = QGuiApplication.screens()

        # If Qt returns an empty list, something is wrong (e.g. no display available),
        # so we can't capture anything.
        if not screens:
            return None

        # Prefer the OS "primary" monitor as the default.
        # As a safety fallback, if primaryScreen() is None for some reason,
        # use the first screen in the list.
        primary = QGuiApplication.primaryScreen() or screens[0]

        # If there is only one monitor, there's nothing to choose from.
        # Just return the default (primary) screen.
        if len(screens) == 1:
            return primary

        # Build a list of human-friendly labels to show in the dropdown dialog.
        labels: list[str] = []

        # Keep a mapping from label -> QScreen so when the user selects a label,
        # we can get back the corresponding QScreen object.
        label_to_screen: dict[str, QScreen] = {}

        # Track which dropdown item should be selected by default.
        # We start with 0, then update it when we see the primary screen.
        primary_index = 0

        # Create one dropdown item per screen.
        # enumerate(..., start=1) makes the labels "Screen 1", "Screen 2", etc.
        for idx, screen in enumerate(screens, start=1):

            # Read the screen geometry so we can display the resolution to the user.
            geometry = screen.geometry()

            # Compose a simple label with:
            # - a user-friendly index
            # - the OS/Qt screen name
            # - the screen resolution (width x height)
            label = f"Screen {idx}: {screen.name()} ({geometry.width()}x{geometry.height()})"

            # If this is the primary screen, mark it in the UI and also remember
            # its position so it shows as the default selection in the dialog.
            if screen == primary:
                label = f"{label} [Primary]"
                primary_index = len(labels)

            # Add the label to the options list shown in the dialog.
            labels.append(label)

            # Store the mapping so we can look up the chosen screen later.
            label_to_screen[label] = screen

        # Show a very simple built-in Qt dialog with a dropdown list.
        # It returns:
        # - selection: the chosen label string
        # - accepted: True if user clicked OK, False if they clicked Cancel/closed it
        selection, accepted = QInputDialog.getItem(
            self,  # parent widget (keeps the dialog centered and modal to this window)
            "Select Screen",  # window title
            "Choose the screen to capture:",  # message text
            labels,  # dropdown items
            primary_index,  # default selected item index
            False,  # editable=False (user must pick one of the options)
        )

        # If the user cancels the dialog, treat it as "no screen selected".
        # The caller can decide what to do (e.g., don't start streaming).
        if not accepted:
            return None

        # Convert the selected label back into the QScreen object.
        # If something unexpected happens (e.g. label not found), fall back to primary.
        return label_to_screen.get(selection, primary)

    # Summary: Slot function that caches the latest frame packet from the proxy.
    # What it does: Stores the packet for the display timer without doing heavy work on the signal.
    # Input: `self`, `packet` (FramePacket).
    # Returns: `None`.
    def _on_bmodeStreamProxy_frame_ready(self, packet: FramePacket) -> None:
        # Cache the packet so rendering can happen on the display timer.
        self._latest_frame_packet = packet
        # Convert the FramePacket fields to the agreed image packet naming.
        image_ts_ms = int(packet.timestamp_ms)
        image_data = packet
        # Cache the latest image packet for consumers that need pull-style access.
        self._latest_image_ts_ms = image_ts_ms
        self._latest_image_data = image_data
        # Emit the packet so the coupling controller can match image+rigidbody samples.
        self.sig_image_packet.emit(image_ts_ms, image_data)
        # Avoid per-frame recording while external coupling is driving snapshots.
        if self._external_recording_active:
            return
        # Enqueue the frame for recording without blocking the UI thread.
        self._image_record_worker.handle_frame(packet)

    # Summary: Slot function that reacts to stream state changes and updates the UI.
    # What it does: Updates buttons/inputs, starts or stops the display timer, and clears the image on stop.
    # Input: `self`, `is_running` (bool), `message` (str).
    # Returns: `None`.
    def _on_bmodeStreamProxy_state_changed(self, is_running: bool, message: str) -> None:
        # The UI has no status text field, so we ignore the message for now.
        _ = message

        if is_running:
            # Update state so the open button toggles to stop.
            self._is_streaming = True
            # UI state change: show stop intent while streaming.
            self.ui.pushButton_bmode_openStream.setText("Stop Stream")
            # UI state change: lock inputs so settings are stable while streaming.
            self.ui.comboBox_bmode_streamOption.setEnabled(False)
            self.ui.comboBox_bmode_streamPort.setEnabled(False)
            self.ui.pushButton_bmode_calibBrowse.setEnabled(False)
            self.ui.pushButton_bmode_calibClear.setEnabled(False)
            self.ui.lineEdit_bmode_calibPath.setEnabled(False)
            # Start the display timer to render cached frames on the UI thread.
            self._display_timer.start()
            return

        # Stop recording if the stream ended so we do not write stale frames.
        if self._image_record_worker.is_active():
            self._stop_recording("Recording stopped because streaming ended.")

        # Update state to idle so the open button toggles back to start.
        self._is_streaming = False
        # UI state change: restore the button label after streaming stops.
        self.ui.pushButton_bmode_openStream.setText("Open Stream")
        # UI state change: re-enable inputs now that streaming is idle.
        self.ui.comboBox_bmode_streamOption.setEnabled(True)
        # Respect the current option when re-enabling the port selector.
        self.ui.comboBox_bmode_streamPort.setEnabled(
            self.ui.comboBox_bmode_streamOption.currentText()
            != "Stream Screen (This PC)"
        )
        self.ui.pushButton_bmode_calibBrowse.setEnabled(True)
        self.ui.pushButton_bmode_calibClear.setEnabled(True)
        self.ui.lineEdit_bmode_calibPath.setEnabled(True)
        # Stop rendering so the UI stays quiet while idle.
        self._display_timer.stop()
        # Clear cached frame data so we do not display stale images.
        self._latest_frame_packet = None
        self._latest_image_ts_ms = None
        self._latest_image_data = None
        self.ui.label_bmode_image.clear()

    # Summary: Slot function that shows worker errors on the UI thread.
    # What it does: Displays a warning dialog and stops streaming cleanly.
    # Input: `self`, `message` (str).
    # Returns: `None`.
    def _on_bmodeStreamProxy_error_message(self, message: str) -> None:
        # Show the warning on the UI thread so it is safe and visible.
        if message:
            QMessageBox.warning(self, "Stream Error", message)
        # Stop streaming so the UI resets after an error.
        self._stop_stream()

    # Summary: Slot function that stops recording when the worker reports a failure.
    # What it does: Stops the writer thread on the UI thread and surfaces the reason once.
    # Input: `self`, `reason` (str).
    # Returns: `None`.
    def _on_bmodeStreamProxy_record_stop(self, reason: str) -> None:
        # Stop recording safely on the UI thread when an error or overload occurs.
        if self._image_record_worker.is_active():
            self._stop_recording(reason)
        elif reason:
            # Still surface the message even if recording already stopped.
            QMessageBox.warning(self, "Recording Stopped", reason)

    # Summary: Slot function that surfaces non-fatal recording messages on the UI thread.
    # What it does: Stores the latest message as a tooltip so the UI stays non-blocking.
    # Input: `self`, `message` (str).
    # Returns: `None`.
    def _on_bmodeStreamProxy_record_message(self, message: str) -> None:
        # Avoid modal dialogs for transient recording warnings.
        if message:
            self.ui.pushButton_bmode_recordStream.setToolTip(message)

    # Summary: Slot function that renders the latest cached frame on a timer.
    # What it does: Delegates to the render helper so heavy work stays on a controlled interval.
    # Input: `self`.
    # Returns: `None`.
    def _on_bmodeDisplayTimer_timeout(self) -> None:
        # Render the latest cached packet on the UI thread.
        self._render_latest_frame()

    # Summary: Render the cached frame packet into the image label.
    # What it does: Builds a QImage from raw bytes, detaches the memory, and displays it.
    # Input: `self`.
    # Returns: `None`.
    def _render_latest_frame(self) -> None:
        # Skip rendering when no packet has arrived yet.
        if self._latest_frame_packet is None:
            return

        packet = self._latest_frame_packet
        # Default to grayscale rendering because the stream payload is now single-channel uint8.
        qimage_format = QImage.Format_Grayscale8
        bytes_per_line = packet.width
        expected_size = bytes_per_line * packet.height

        # Keep RGB fallback support so older/legacy packets still display correctly.
        if packet.fmt == "rgb888":
            qimage_format = QImage.Format_RGB888
            bytes_per_line = packet.width * 3
            expected_size = bytes_per_line * packet.height

        if len(packet.data) < expected_size:
            # Ignore incomplete packets to avoid rendering garbage.
            return

        # Create a QImage from the raw bytes using the packet format.
        qimage = QImage(
            packet.data,
            packet.width,
            packet.height,
            bytes_per_line,
            qimage_format,
        )
        # Detach the QImage from the raw buffer so it stays valid after this method.
        qimage = qimage.copy()
        pixmap = QPixmap.fromImage(qimage)
        self._display_pixmap(pixmap)

    # Summary: Show a pixmap in the UI label.
    # What it does: Scales the pixmap to fit the label while keeping the aspect ratio.
    # Input: `self`, `pixmap` (the image to show).
    # Returns: `None`.
    def _display_pixmap(self, pixmap: QPixmap) -> None:
        # Scale the pixmap to the label with the same style as camera frames.
        scaled = pixmap.scaled(
            self.ui.label_bmode_image.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.ui.label_bmode_image.setPixmap(scaled)

    # Summary: Stop streaming and reset the UI.
    # What it does: Stops workers and timers, clears cached frames, and restores UI controls.
    # Input: `self`.
    # Returns: `None`.
    def _stop_stream(self) -> None:
        # Stop the workers so acquisition ends in both camera and screen paths.
        self._camera_worker.stop()
        self._screen_worker.stop()
        # Stop rendering so the UI does not redraw stale frames.
        self._display_timer.stop()
        # Reset the stream state so the open/stop button toggles correctly.
        self._is_streaming = False
        # UI state change: restore the button label after stopping.
        self.ui.pushButton_bmode_openStream.setText("Open Stream")
        # UI state change: re-enable inputs now that streaming has stopped.
        self.ui.comboBox_bmode_streamOption.setEnabled(True)
        self.ui.comboBox_bmode_streamPort.setEnabled(
            self.ui.comboBox_bmode_streamOption.currentText()
            != "Stream Screen (This PC)"
        )
        self.ui.pushButton_bmode_calibBrowse.setEnabled(True)
        self.ui.pushButton_bmode_calibClear.setEnabled(True)
        self.ui.lineEdit_bmode_calibPath.setEnabled(True)
        # Clear cached frame data and the image label.
        self._latest_frame_packet = None
        self._latest_image_ts_ms = None
        self._latest_image_data = None
        self.ui.label_bmode_image.clear()

    # Summary: Handle the window close event.
    # What it does: Stops streaming so workers and timers are cleaned up before the app closes.
    # Input: `self`, `event` (the Qt close event object).
    # Returns: `None`.
    def closeEvent(self, event) -> None:
        # Stop recording so the writer thread can close cleanly.
        if self._image_record_worker.is_active():
            self._stop_recording("Recording stopped because the window closed.")
        # Ensure worker threads and timers are stopped when the window closes.
        self._stop_stream()
        # Call the base class close handler after our cleanup.
        super().closeEvent(event)
        event.accept()


# Summary: App entry point for the B-mode streaming UI.
# What it does: Creates the QApplication, shows the main window, and starts the Qt event loop.
# Input: No arguments; reads `sys.argv` for Qt options.
# Returns: `None` (the process exits via `sys.exit(...)`).
def main() -> None:
    """Create the QApplication and show the V2 window."""
    # Create the Qt application and pass through any command-line args.
    app = QApplication(sys.argv)
    # Instantiate and show the main window that renders the V2 form.
    window = BModeWidget()
    window.show()
    # Enter the Qt event loop and exit the process when it ends.
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
