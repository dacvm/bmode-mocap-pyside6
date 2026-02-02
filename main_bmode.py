
"""Live-preview widget that streams from the selected USB camera into the Qt UI."""

# Standard library helpers for launching a Qt application and accepting CLI args.
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
# Optional typing helper preserved for future extensions or annotations.
from typing import Optional

# Third-party libs for video capture (OpenCV) and Qt widgets/layouts.
import cv2
from PySide6.QtCore import QObject, Signal, QTimer, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPixmap, QScreen
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox, QWidget

# Generated UI classes that wrap the Qt Designer forms into Python-friendly fields.
from bmode_ui import Ui_Form as Ui_BModeV2
# Class to parse the Calibration XML.
from helpers.xml_stream_parser import (
    CoordinateDefinitionsExtractor,
    VideoDeviceExtractor,
    XmlStreamParser,
)


# Summary:
# - Build a UTC epoch timestamp in milliseconds for frame packets.
# - What it does: Uses time.time() to produce an integer ms value that is safe for threading.
# - Input: None.
# - Returns: Milliseconds since Unix epoch (int).
def _utc_epoch_ms() -> int:
    # Use time.time because it is lightweight and thread-safe for this use case.
    return int(time.time() * 1000)


# Summary:
# - Lightweight container for a single RGB frame and its metadata.
# - What it does: Stores timestamp, dimensions, format label, and raw RGB bytes for UI rendering.
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

                # Convert OpenCV's BGR to RGB for Qt rendering.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _channels = rgb.shape

                # Package the frame so the UI thread can render safely.
                packet = FramePacket(
                    _utc_epoch_ms(),
                    width,
                    height,
                    "rgb888",
                    rgb.tobytes(),
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

        # Convert the pixmap into a packed RGB888 byte buffer.
        qimg = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        bytes_per_line = qimg.bytesPerLine()
        expected_bytes_per_line = qimg.width() * 3
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
            # Remove per-line padding so the payload is tightly packed RGB888.
            packed_lines = []
            for row in range(height):
                start = row * bytes_per_line
                packed_lines.append(raw[start : start + expected_bytes_per_line])
            data = b"".join(packed_lines)
        else:
            data = raw

        # Emit the frame packet for UI rendering.
        packet = FramePacket(
            _utc_epoch_ms(),
            qimg.width(),
            qimg.height(),
            "rgb888",
            data,
        )
        self._proxy.frame_ready.emit(packet)


# Summary: Main Qt window for the B-mode streaming UI.
# What it does: Builds the widgets from the Designer `.ui`, wires signals, and manages stream state.
# Input: Created with no args; uses `self.ui` widgets and internal state to react to user actions.
# Returns: A QWidget subclass instance that can be shown in a QApplication.
class BModeWidget(QWidget):
    """Lightweight window that simply renders the V2 form."""

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

        # Cache the most recent frame packet so the UI can render on a timer.
        self._latest_frame_packet: Optional[FramePacket] = None
        # Track the current stream state to keep the open/stop button repeat-safe.
        self._is_streaming = False

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
        bytes_per_line = packet.width * 3
        expected_size = bytes_per_line * packet.height
        if len(packet.data) < expected_size:
            # Ignore incomplete packets to avoid rendering garbage.
            return

        # Create a QImage from the raw RGB bytes.
        qimage = QImage(
            packet.data,
            packet.width,
            packet.height,
            bytes_per_line,
            QImage.Format_RGB888,
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
        self.ui.label_bmode_image.clear()

    # Summary: Handle the window close event.
    # What it does: Stops streaming so workers and timers are cleaned up before the app closes.
    # Input: `self`, `event` (the Qt close event object).
    # Returns: `None`.
    def closeEvent(self, event) -> None:
        # Ensure worker threads and timers are stopped when the window closes.
        self._stop_stream()
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
