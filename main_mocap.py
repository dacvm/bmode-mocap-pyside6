"""PySide6 widget to stream Qualisys 6D residual data without blocking the UI."""

# Standard library imports for async streaming, background threads, and numeric helpers.
import asyncio
import csv
import ipaddress
import math
import os
import queue
import threading
import time
import xml.etree.ElementTree as ElementTree
from datetime import datetime, timezone
from typing import Optional

# Third-party imports for QTM real-time streaming, numeric transforms, plotting, and Qt UI.
import qtm_rt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

# Generated UI class from Qt Designer (do not edit the _ui.py file).
from mocap_ui import Ui_Form as Ui_Mocap

QTM_PORT = 22223
QTM_PROTOCOL_VERSION = "1.20"

# Recording constants to keep the writer thread responsive and avoid unbounded memory use.
RECORD_QUEUE_MAXSIZE = 500
RECORD_FLUSH_EVERY = 50
RECORD_DROP_WARN_INTERVAL = 1.0
RECORD_DROP_WINDOW_SECONDS = 2.0
RECORD_DROP_RATE_THRESHOLD = 0.05
RECORD_NAN = float("nan")


# Summary:
# - Build a monotonic timestamp in milliseconds for mocap packets.
# - What it does: Uses `time.monotonic()` to keep rigid body timestamps on the same
#   stable clock source as image timestamps used for coupling.
# - Input: None.
# - Returns: Monotonic milliseconds (int).
def _now_ms() -> int:
    # Use a monotonic clock so packet deltas stay valid if wall clock time changes.
    return int(time.monotonic() * 1000)


# Summary:
# - Matplotlib canvas that renders a live 3D scatter plot for mocap body positions.
# - What it does: owns the Figure, a 3D Axes, and a single scatter handle that is updated in place.
class Mocap3DCanvas(FigureCanvas):
    # Base transform to map QTM's Y-up world into Matplotlib's Z-up world.
    T_base = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    # Summary:
    # - Initialize the 3D plot with labels and an empty scatter handle.
    # - Input: `self`, `parent` (QWidget | None).
    # - Returns: None.
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        # Build the figure and axes once so we can update without clearing every frame.
        figure = Figure()
        axes = figure.add_subplot(111, projection="3d")
        super().__init__(figure)

        # Keep a reference to the axes for fast updates later.
        self._axes = axes

        # Create the scatter handle once so updates only move existing points.
        self._scatter = self._axes.scatter([], [], [], s=20)

        # Track text labels so each point can show its rigid body name.
        self._label_texts: list = []

        # Disable autoscale so camera stays stable after we set limits once.
        self._axes.set_autoscale_on(False)

        # Label axes so users know the coordinate directions (Z is up after the base transform).
        self._axes.set_xlabel("X")
        self._axes.set_ylabel("Y")
        self._axes.set_zlabel("Z (Up)")
        self._axes.set_aspect("equal")
        self._axes.set_proj_type("ortho")

        # Track whether we already autoscaled to the first valid frame.
        self._has_autoscaled = False

        # Ensure the canvas is parented to the widget tree if provided.
        if parent is not None:
            self.setParent(parent)

    # Summary:
    # - Ensure the label text artists match the number of rigid body labels.
    # - Input: `self`, `labels` (list[str]).
    # - Returns: None.
    def _sync_label_artists(self, labels: list[str]) -> None:
        # Remove extra label artists so stale labels are not shown.
        while len(self._label_texts) > len(labels):
            text_artist = self._label_texts.pop()
            # Remove from the axes so it is no longer drawn.
            text_artist.remove()

        # Add missing label artists so every body has a text placeholder.
        while len(self._label_texts) < len(labels):
            label = labels[len(self._label_texts)]
            # Create the text in a neutral location; we move it per-frame.
            # Create text without a fixed 3D direction so it stays horizontal.
            text_artist = self._axes.text(0.0, 0.0, 0.0, label, fontsize=9, rotation=0)
            self._label_texts.append(text_artist)

    # Summary:
    # - Update the scatter points using the latest QTM poses.
    # - Input: `self`, `poses_qtm` (dict[str, np.ndarray | None]).
    # - Returns: None.
    def update_poses(self, poses_qtm: dict[str, Optional[np.ndarray]]) -> None:
        # Skip updates when no poses have been cached yet.
        if poses_qtm is None:
            return

        # Keep label order aligned with dict insertion order so points and names match.
        labels = list(poses_qtm.keys())
        # Sync label artists with the incoming labels before updating positions.
        self._sync_label_artists(labels)

        x_values: list[float] = []
        y_values: list[float] = []
        z_values: list[float] = []
        valid_positions: list[tuple[float, float, float]] = []

        # Convert each pose into scatter arrays while keeping insertion order.
        for index, (label, pose_qtm) in enumerate(poses_qtm.items()):
            # Grab the matching text artist so each point can show its label.
            label_artist = self._label_texts[index]

            # Treat missing entries as NaN so Matplotlib skips them.
            if pose_qtm is None or not isinstance(pose_qtm, np.ndarray):
                x_values.append(math.nan)
                y_values.append(math.nan)
                z_values.append(math.nan)
                # Hide the label because there is no visible point.
                label_artist.set_visible(False)
                continue

            # Ensure we only process valid homogeneous transforms.
            if pose_qtm.shape != (4, 4):
                x_values.append(math.nan)
                y_values.append(math.nan)
                z_values.append(math.nan)
                # Hide the label because the pose is not drawable.
                label_artist.set_visible(False)
                continue

            # Apply the fixed base transform (left-multiplication changes world frame to plot frame).
            pose_plot = self.T_base @ pose_qtm
            x_value, y_value, z_value = pose_plot[:3, 3]
            x_values.append(float(x_value))
            y_values.append(float(y_value))
            z_values.append(float(z_value))

            # Use finite values only when we compute the initial autoscale.
            if (
                math.isfinite(x_value)
                and math.isfinite(y_value)
                and math.isfinite(z_value)
            ):
                valid_positions.append((float(x_value), float(y_value), float(z_value)))
                # Update the label position so it follows the visible point.
                label_artist.set_text(label)
                label_artist.set_position((float(x_value), float(y_value)))
                # Keep labels horizontal by avoiding a 3D direction alignment.
                label_artist.set_3d_properties(float(z_value), zdir=None)
                label_artist.set_visible(True)
            else:
                # Hide the label when the values are not finite.
                label_artist.set_visible(False)

        # Update the existing scatter in place instead of recreating it.
        self._scatter._offsets3d = (x_values, y_values, z_values)

        # Autoscale once on the first valid frame so the camera stays stable.
        if valid_positions and not self._has_autoscaled:
            # Compute bounds with a small margin so points are not on the edges.
            x_list = [pos[0] for pos in valid_positions]
            y_list = [pos[1] for pos in valid_positions]
            z_list = [pos[2] for pos in valid_positions]

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            z_min, z_max = min(z_list), max(z_list)

            x_margin = max((x_max - x_min) * 0.1, 1.0)
            y_margin = max((y_max - y_min) * 0.1, 1.0)
            z_margin = max((z_max - z_min) * 0.1, 1.0)

            self._axes.set_xlim(x_min - x_margin, x_max + x_margin)
            self._axes.set_ylim(y_min - y_margin, y_max + y_margin)
            self._axes.set_zlim(z_min - z_margin, z_max + z_margin)

            # Lock in the first autoscale to keep the camera stable.
            self._has_autoscaled = True

        # Request a redraw without blocking the UI thread.
        self.draw_idle()

    # Summary:
    # - Clear the scatter points and reset autoscale tracking.
    # - Input: `self`.
    # - Returns: None.
    def clear_points(self) -> None:
        # Empty the scatter data so the plot visibly clears on stop.
        self._scatter._offsets3d = ([], [], [])
        # Hide labels so the cleared plot has no lingering text.
        for label_artist in self._label_texts:
            label_artist.set_visible(False)
        # Allow autoscaling again when the next stream starts.
        self._has_autoscaled = False
        # Schedule a redraw so the clear is visible to the user.
        self.draw_idle()


# Summary:
# - Qt signal bridge used to safely move data/state from the asyncio streaming thread into the Qt UI thread.
# - Why it exists: Qt widgets must only be updated on the main thread, but QTM packets arrive on a background
#   thread. This proxy provides Qt 'Signals' that the widget connects to, so updates are queued and delivered
#   in a thread-safe way.
class MocapStreamProxy(QObject):
    text_ready = Signal(str)
    poses_ready = Signal(int, object)
    state_changed = Signal(bool, str)
    record_message = Signal(str)
    record_stop = Signal(str)

    # Summary:
    # - Initialize the proxy and assign a stable objectName for slot naming consistency.
    # - Input: `self`.
    # - Returns: None.
    def __init__(self) -> None:
        super().__init__()
        # Give the proxy a name so slot handlers can follow the naming convention.
        self.setObjectName("mocapStreamProxy")


# Summary:
# - Worker that owns CSV recording state and a background writer thread.
# - What it does: receives 6D residual results, formats rows, and writes them to disk without blocking.
class CsvRecordWorker:
    # Summary:
    # - Build a UTC epoch timestamp in milliseconds for CSV rows.
    # - Input: None.
    # - Returns: Milliseconds since Unix epoch at UTC (int).
    @staticmethod
    def _utc_epoch_ms() -> int:
        # Use UTC time so files are consistent across machines and time zones.
        now_utc = datetime.now(timezone.utc)
        # Convert to integer milliseconds to avoid floating-point timestamps.
        return int(now_utc.timestamp() * 1000)

    # Summary:
    # - Normalize a quaternion to unit length and provide a safe fallback.
    # - Input: `quat` (tuple[float, float, float, float]).
    # - Returns: Normalized quaternion as (w, x, y, z) (tuple[float, float, float, float]).
    @staticmethod
    def _normalize_quaternion(
        quat: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        # Compute the norm so we can avoid division by zero.
        w_value, x_value, y_value, z_value = quat
        norm = math.sqrt(
            (w_value * w_value)
            + (x_value * x_value)
            + (y_value * y_value)
            + (z_value * z_value)
        )
        if norm <= 0.0:
            # Fall back to identity when the norm is invalid.
            return (1.0, 0.0, 0.0, 0.0)
        return (w_value / norm, x_value / norm, y_value / norm, z_value / norm)

    # Summary:
    # - Convert a 3x3 rotation matrix into a stable quaternion (w, x, y, z).
    # - Input: `rotation_3x3` (list[list[float]]).
    # - Returns: Quaternion tuple (w, x, y, z) (tuple[float, float, float, float]).
    @staticmethod
    def _rotation_matrix_to_quaternion(
        rotation_3x3: list[list[float]],
    ) -> tuple[float, float, float, float]:
        # Use the classic stable branch selection based on the matrix trace.
        m00, m01, m02 = rotation_3x3[0]
        m10, m11, m12 = rotation_3x3[1]
        m20, m21, m22 = rotation_3x3[2]

        trace = m00 + m11 + m22
        if trace > 0.0:
            # Trace-positive path minimizes numerical error when the angle is small.
            scale = math.sqrt(trace + 1.0) * 2.0
            w_value = 0.25 * scale
            x_value = (m21 - m12) / scale
            y_value = (m02 - m20) / scale
            z_value = (m10 - m01) / scale
        elif m00 > m11 and m00 > m22:
            # Diagonal-major path for X dominance.
            scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            w_value = (m21 - m12) / scale
            x_value = 0.25 * scale
            y_value = (m01 + m10) / scale
            z_value = (m02 + m20) / scale
        elif m11 > m22:
            # Diagonal-major path for Y dominance.
            scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            w_value = (m02 - m20) / scale
            x_value = (m01 + m10) / scale
            y_value = 0.25 * scale
            z_value = (m12 + m21) / scale
        else:
            # Diagonal-major path for Z dominance.
            scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            w_value = (m10 - m01) / scale
            x_value = (m02 + m20) / scale
            y_value = (m12 + m21) / scale
            z_value = 0.25 * scale

        # Normalize to unit length to keep downstream math stable.
        return CsvRecordWorker._normalize_quaternion(
            (w_value, x_value, y_value, z_value)
        )

    # Summary:
    # - Initialize the CSV recording worker and its state containers.
    # - Input: `self`, `proxy` (MocapStreamProxy).
    # - Returns: None.
    def __init__(self, proxy: MocapStreamProxy) -> None:
        # Store the proxy so background threads can send UI-safe signals.
        self._proxy = proxy

        # Track whether recording is active so we can ignore packets when stopped.
        self._active = False
        # Store thread resources that are created on start and released on stop.
        self._queue: Optional[queue.Queue] = None
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

        # Persist the body ordering for CSV columns and quaternion continuity.
        self._record_body_names: list[str] = []
        self._record_prev_quaternion: dict[str, tuple[float, float, float, float]] = {}

        # Track drop statistics to warn and auto-stop on sustained overload.
        self._record_recent_drop_count = 0
        self._record_recent_drop_rate = 0.0
        self._record_drop_window_start: Optional[float] = None
        self._record_drop_window_total = 0
        self._record_drop_window_dropped = 0
        self._record_drop_total = 0
        self._record_last_drop_warning_time = 0.0
        self._record_overload_triggered = False

        # Guard start/stop so we do not race if the UI toggles quickly.
        self._lock = threading.Lock()

    # Summary:
    # - Start the CSV writer thread and reset recording state for a new session.
    # - Input: `self`, `file_path` (str), `header` (list[str]), `body_names` (list[str]).
    # - Returns: None.
    def start(self, file_path: str, header: list[str], body_names: list[str]) -> None:
        # Ensure only one recording session is active at a time.
        with self._lock:
            if self._active:
                return
            self._active = True

            # Freeze the body names and reset per-session state.
            self._record_body_names = list(body_names)
            self._record_prev_quaternion = {}
            self._record_recent_drop_count = 0
            self._record_recent_drop_rate = 0.0
            self._record_drop_window_start = None
            self._record_drop_window_total = 0
            self._record_drop_window_dropped = 0
            self._record_drop_total = 0
            self._record_last_drop_warning_time = 0.0
            self._record_overload_triggered = False

            # Create fresh queue/stop event per session so writer state is isolated.
            self._queue = queue.Queue(maxsize=RECORD_QUEUE_MAXSIZE)
            self._stop_event = threading.Event()

            # Start the writer thread so file I/O never blocks the stream thread.
            self._thread = threading.Thread(
                target=self._writer_worker,
                args=(file_path, header, self._queue, self._stop_event),
                daemon=True,
            )
            # Thread start: begin draining rows to disk asynchronously.
            self._thread.start()

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
            self._record_body_names = []
            self._record_prev_quaternion = {}

        # Signal the writer thread to drain and exit.
        if stop_event is not None:
            stop_event.set()

        # Join briefly so we do not freeze the UI while final rows flush.
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
    # - Handle a 6D residual result by enqueuing a CSV row when recording is active.
    # - Input: `self`, `result` (tuple | None).
    # - Returns: None.
    def handle_6d_residual_result(self, result) -> None:
        # Snapshot state under lock so stop/start does not race packet handling.
        with self._lock:
            if not self._active:
                return
            record_queue = self._queue
            record_body_names = list(self._record_body_names)
            stop_event = self._stop_event
        if (
            record_queue is None
            or not record_body_names
            or (stop_event is not None and stop_event.is_set())
        ):
            return

        # Capture the arrival time on the stream thread for accurate timestamps.
        row = [self._utc_epoch_ms()]

        # Fill NaNs when the 6D component is missing in the current packet.
        if result is None:
            row.extend([RECORD_NAN] * (len(record_body_names) * 7))
        else:
            _header, bodies = result
            for index, body_name in enumerate(record_body_names):
                # Guard against packets with fewer bodies than expected.
                if index >= len(bodies):
                    row.extend([RECORD_NAN] * 7)
                    continue

                body = bodies[index]
                if body is None:
                    row.extend([RECORD_NAN] * 7)
                    continue

                # Extract position and rotation while ignoring residuals.
                try:
                    position, rotation, _residual = body
                except (TypeError, ValueError):
                    row.extend([RECORD_NAN] * 7)
                    continue

                if position is None or rotation is None:
                    row.extend([RECORD_NAN] * 7)
                    continue

                # Read translation in raw QTM coordinates (no plot transform).
                if hasattr(position, "x") and hasattr(position, "y") and hasattr(
                    position, "z"
                ):
                    x_value, y_value, z_value = position.x, position.y, position.z
                else:
                    try:
                        x_value, y_value, z_value = position
                    except (TypeError, ValueError):
                        row.extend([RECORD_NAN] * 7)
                        continue

                try:
                    x_float = float(x_value)
                    y_float = float(y_value)
                    z_float = float(z_value)
                except (TypeError, ValueError):
                    row.extend([RECORD_NAN] * 7)
                    continue

                # Pull the rotation matrix and convert it to a quaternion.
                rotation_matrix = getattr(rotation, "matrix", None)
                if rotation_matrix is None:
                    row.extend([RECORD_NAN] * 7)
                    continue

                try:
                    rotation_values = [float(value) for value in rotation_matrix]
                except (TypeError, ValueError):
                    row.extend([RECORD_NAN] * 7)
                    continue
                if len(rotation_values) != 9:
                    row.extend([RECORD_NAN] * 7)
                    continue

                rotation_3x3 = [
                    rotation_values[0:3],
                    rotation_values[3:6],
                    rotation_values[6:9],
                ]
                quat = self._rotation_matrix_to_quaternion(rotation_3x3)

                # Optional sign continuity to avoid sudden sign flips per body.
                prev_quat = self._record_prev_quaternion.get(body_name)
                if prev_quat is not None:
                    dot_value = (
                        (prev_quat[0] * quat[0])
                        + (prev_quat[1] * quat[1])
                        + (prev_quat[2] * quat[2])
                        + (prev_quat[3] * quat[3])
                    )
                    if dot_value < 0.0:
                        quat = (-quat[0], -quat[1], -quat[2], -quat[3])

                self._record_prev_quaternion[body_name] = quat
                row.extend(
                    [quat[0], quat[1], quat[2], quat[3], x_float, y_float, z_float]
                )

        # Enqueue the row without blocking so the stream thread stays responsive.
        did_drop = False
        try:
            record_queue.put_nowait(row)
        except queue.Full:
            # Drop newest when the queue is full to protect latency.
            did_drop = True

        now_monotonic = time.monotonic()
        overload = self._update_record_drop_stats(did_drop, now_monotonic)

        # Rate-limit UI warnings so we do not spam the text area.
        if did_drop and (
            now_monotonic - self._record_last_drop_warning_time
            >= RECORD_DROP_WARN_INTERVAL
        ):
            self._record_last_drop_warning_time = now_monotonic
            message = (
                f"Recording queue full: dropped {self._record_drop_total} frames total."
            )
            if self._record_recent_drop_rate > 0.0:
                message += (
                    f" Recent drop rate {self._record_recent_drop_rate * 100.0:.1f}%."
                )
            self._safe_emit(self._proxy.record_message.emit, message)

        # Auto-stop recording when sustained overload exceeds the threshold.
        if overload and not self._record_overload_triggered:
            self._record_overload_triggered = True
            self._safe_emit(
                self._proxy.record_stop.emit,
                "Recording stopped: disk too slow or queue overflow "
                f"(dropped {self._record_recent_drop_count} frames in "
                f"{RECORD_DROP_WINDOW_SECONDS:.0f}s; total dropped {self._record_drop_total})."
            )

    # Summary:
    # - Update the drop counters and decide whether sustained overload occurred.
    # - Input: `self`, `did_drop` (bool), `now_monotonic` (float).
    # - Returns: True if overload threshold is exceeded, else False (bool).
    def _update_record_drop_stats(self, did_drop: bool, now_monotonic: float) -> bool:
        # Initialize the drop window on the first packet.
        if self._record_drop_window_start is None:
            self._record_drop_window_start = now_monotonic
            self._record_drop_window_total = 0
            self._record_drop_window_dropped = 0

        # Track total samples for the current window.
        self._record_drop_window_total += 1
        if did_drop:
            # Track dropped samples and overall dropped count.
            self._record_drop_window_dropped += 1
            self._record_drop_total += 1

        # Only evaluate overload once the window duration has elapsed.
        elapsed = now_monotonic - self._record_drop_window_start
        if elapsed < RECORD_DROP_WINDOW_SECONDS:
            return False

        # Compute recent metrics from the finished window.
        if self._record_drop_window_total > 0:
            self._record_recent_drop_count = self._record_drop_window_dropped
            self._record_recent_drop_rate = (
                self._record_drop_window_dropped / self._record_drop_window_total
            )
        else:
            self._record_recent_drop_count = 0
            self._record_recent_drop_rate = 0.0

        # Reset the window counters for the next evaluation cycle.
        self._record_drop_window_start = now_monotonic
        self._record_drop_window_total = 0
        self._record_drop_window_dropped = 0

        # Return overload state so callers can auto-stop when needed.
        return self._record_recent_drop_rate > RECORD_DROP_RATE_THRESHOLD

    # Summary:
    # - Writer thread entry point that owns file I/O for CSV recording.
    # - Input: `self`, `file_path` (str), `header` (list[str]),
    #   `record_queue` (queue.Queue), `stop_event` (threading.Event).
    # - Returns: None.
    def _writer_worker(
        self,
        file_path: str,
        header: list[str],
        record_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        # Summary:
        # - Format numeric values for CSV output with fixed decimal precision.
        # - Input: `value` (object).
        # - Returns: Formatted value for CSV output (object).
        def _format_csv_value(value):
            # Keep integers unchanged so timestamps remain exact.
            if isinstance(value, int):
                return value
            # Format floats with up to 4 decimals and keep NaN readable.
            if isinstance(value, float):
                if math.isnan(value):
                    return "NaN"
                return f"{value:.4f}"
            return value

        # Wrap file I/O in a try/except so writer errors do not crash the app.
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                # Write the header once at the top of the file.
                writer.writerow(header)

                rows_since_flush = 0
                while True:
                    if stop_event.is_set():
                        # Drain any remaining rows before exiting.
                        try:
                            row = record_queue.get_nowait()
                        except queue.Empty:
                            break
                    else:
                        # Wait briefly for new rows so we do not spin.
                        try:
                            row = record_queue.get(timeout=0.1)
                        except queue.Empty:
                            continue

                    # Format floats on the writer thread to keep the stream thread light.
                    formatted_row = [_format_csv_value(value) for value in row]
                    writer.writerow(formatted_row)
                    rows_since_flush += 1
                    # Flush periodically so data hits disk without per-row overhead.
                    if rows_since_flush >= RECORD_FLUSH_EVERY:
                        csv_file.flush()
                        rows_since_flush = 0

                # Flush one last time before closing the file.
                csv_file.flush()
        except Exception as exc:
            # Report writer failures back to the UI without touching widgets directly.
            self._safe_emit(self._proxy.record_stop.emit, f"Recording stopped: {exc}")


# Summary:
# - Worker that owns the asyncio loop and QTM streaming thread.
# - What it does: connects to QTM, streams packets, and forwards data through MocapStreamProxy.
class QtmStreamWorker:
    # Summary:
    # - Initialize the stream worker with its proxy and recorder references.
    # - Input: `self`, `proxy` (MocapStreamProxy), `recorder` (CsvRecordWorker).
    # - Returns: None.
    def __init__(self, proxy: MocapStreamProxy, recorder: CsvRecordWorker) -> None:
        # Store the proxy to emit UI-safe signals from the stream thread.
        self._proxy = proxy
        # Store the recorder so packet callbacks can forward results.
        self._recorder = recorder

        # Track the stream thread and asyncio loop for safe stopping.
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None

        # Cache body names so packet formatting uses stable labels.
        self._body_names: list[str] = []

        # Lock protects loop/body name updates across threads.
        self._lock = threading.Lock()

    # Summary:
    # - Parse the 6D parameters XML and return a list of rigid body names.
    # - Input: `xml_string` (str).
    # - Returns: A list of body names (list[str]).
    @staticmethod
    def _parse_6d_body_names(xml_string: str) -> list[str]:
        # Return early when there is no XML, so callers can handle missing data safely.
        if not xml_string:
            return []

        # Parse the XML and guard against malformed data that could crash the stream.
        try:
            root = ElementTree.fromstring(xml_string)
        except ElementTree.ParseError:
            return []

        # Find the 6D section because body names live inside The_6D/Body/Name nodes.
        the_6d = root.find(".//The_6D")
        if the_6d is None:
            return []

        # Collect body names in the order QTM reports them.
        body_names: list[str] = []
        for body in the_6d.findall("Body"):
            name_element = body.find("Name")
            if name_element is not None and name_element.text:
                body_names.append(name_element.text.strip())

        return body_names

    # Summary:
    # - Request 6D parameters from QTM so we can map rigid body indices to names.
    # - Input: `connection` (qtm_rt.QRTConnection).
    # - Returns: A list of rigid body names (list[str]).
    @staticmethod
    async def _load_body_names(connection) -> list[str]:
        # Ask QTM for its 6D parameter XML so the packet formatter can show labels.
        xml_string = await connection.get_parameters(parameters=["6d"])
        # Transform the XML into a simple list of names.
        return QtmStreamWorker._parse_6d_body_names(xml_string)

    # Summary:
    # - Convert a 6D residual packet into a readable multi-line string.
    # - Input: `packet` (qtm_rt.QRTPacket), `body_names` (list[str]), `result` (tuple | None).
    # - Returns: Formatted text for display in the UI (str).
    @staticmethod
    def _format_6d_residual_text(packet, body_names: list[str], result=None) -> str:
        # Use a provided 6D result so callers can avoid duplicate packet reads.
        if result is None:
            # Always check for missing components because QTM can send empty frames.
            result = packet.get_6d_residual()
        if result is None:
            return f"Frame {packet.framenumber} | No 6D residual data"

        # Unpack the header and body list now that we know the component exists.
        header, bodies = result
        lines = [f"Frame {packet.framenumber} | {header}"]

        # Build a line per rigid body with position, rotation matrix, and residual.
        for index, body in enumerate(bodies):
            # Map the body index to a known name when possible.
            label = body_names[index] if index < len(body_names) else f"Body{index + 1}"
            position, rotation, residual = body

            # Extract position coordinates and guard against missing tuples.
            x, y, z = position if position is not None else (0.0, 0.0, 0.0)
            # Pull the rotation matrix if present; otherwise mark it as missing.
            rotation_matrix = getattr(rotation, "matrix", None)
            # Pull the residual value if present; otherwise mark it as missing.
            residual_value = getattr(residual, "residual", None)

            if rotation_matrix is None or residual_value is None:
                # Provide a safe fallback when QTM omits rotation/residual data.
                lines.append(
                    f"{label}: X={x:.2f}, Y={y:.2f}, Z={z:.2f} | R=N/A | Residual=N/A"
                )
                continue

            r11, r12, r13, r21, r22, r23, r31, r32, r33 = rotation_matrix
            lines.append(
                f"{label}:\n"
                f"X = [ {x:.2f}, {y:.2f}, {z:.2f}]\n"
                f"R = [ {r11:.2f}  {r12:.2f}  {r13:.2f};\n"
                f"      {r21:.2f}  {r22:.2f}  {r23:.2f};\n"
                f"      {r31:.2f}  {r32:.2f}  {r33:.2f}]\n"
                f"Res = {residual_value:.2f}"
            )

        return "\n".join(lines)

    # Summary:
    # - Build a dict of 6D rigid body poses aligned to the configured body names.
    # - Input: `result` (tuple | None), `body_names` (list[str]).
    # - Returns: Mapping of body name to 4x4 pose matrix or None (dict[str, np.ndarray | None]).
    @staticmethod
    def _extract_6d_poses(
        result, body_names: list[str]
    ) -> dict[str, Optional[np.ndarray]]:
        # Return an empty mapping when there is no data and no known bodies.
        if result is None and not body_names:
            return {}

        # If there is no 6D component, return None for each known body name.
        if result is None:
            return {name: None for name in body_names}

        # Unpack the 6D result safely; the header is unused for pose extraction.
        _header, bodies = result

        # Build a stable list of names so dict insertion order stays consistent.
        if body_names:
            target_names = body_names
        else:
            # Fall back to generic names when QTM parameters are unavailable.
            target_names = [f"Body{index + 1}" for index in range(len(bodies))]

        poses: dict[str, Optional[np.ndarray]] = {}
        for index, name in enumerate(target_names):
            # Store None when QTM omits a body so downstream plotting can skip it safely.
            if index >= len(bodies):
                poses[name] = None
                continue

            body = bodies[index]
            # Guard against empty body tuples to avoid crashes.
            if body is None:
                poses[name] = None
                continue

            # The SDK returns (position, rotation, residual); we need position + rotation.
            try:
                position, rotation, _residual = body
            except (TypeError, ValueError):
                poses[name] = None
                continue

            # Skip missing components because a full pose needs both translation and rotation.
            if position is None or rotation is None:
                poses[name] = None
                continue

            # Support tuple-style positions; fall back to attributes if provided.
            if hasattr(position, "x") and hasattr(position, "y") and hasattr(
                position, "z"
            ):
                x_value, y_value, z_value = position.x, position.y, position.z
            else:
                try:
                    x_value, y_value, z_value = position
                except (TypeError, ValueError):
                    poses[name] = None
                    continue

            # Convert values to floats; if conversion fails, store None.
            try:
                x_float = float(x_value)
                y_float = float(y_value)
                z_float = float(z_value)
            except (TypeError, ValueError):
                poses[name] = None
                continue

            # Pull the rotation matrix from the SDK rotation object.
            rotation_matrix = getattr(rotation, "matrix", None)
            if rotation_matrix is None:
                poses[name] = None
                continue

            # Convert rotation entries to floats and validate the expected length.
            try:
                rotation_values = [float(value) for value in rotation_matrix]
            except (TypeError, ValueError):
                poses[name] = None
                continue
            if len(rotation_values) != 9:
                poses[name] = None
                continue

            # QTM delivers r11..r33 in row-major order; reshape in C-order for consistency.
            rotation_3x3 = np.array(rotation_values, dtype=float).reshape((3, 3))

            # Assemble the homogeneous transform in QTM's raw coordinate frame.
            pose_matrix = np.eye(4, dtype=float)
            pose_matrix[:3, :3] = rotation_3x3
            pose_matrix[:3, 3] = [x_float, y_float, z_float]

            poses[name] = pose_matrix

        return poses

    # Summary:
    # - Start the QTM streaming thread for the given IP address.
    # - Input: `self`, `ip_address` (str).
    # - Returns: None.
    def start(self, ip_address: str) -> None:
        # Avoid launching duplicate stream threads.
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return

            # Thread start: run the asyncio loop in a background thread.
            self._thread = threading.Thread(
                target=self._run_stream_loop,
                args=(ip_address,),
                daemon=True,
            )
            self._thread.start()

    # Summary:
    # - Signal the asyncio loop to stop streaming safely.
    # - Input: `self`.
    # - Returns: None.
    def stop(self) -> None:
        # Snapshot the loop and stop event under lock for thread-safe access.
        with self._lock:
            loop = self._loop
            stop_event = self._stop_event

        # Exit early if the loop never started or already stopped.
        if loop is None or stop_event is None:
            return

        # Summary:
        # - Thread-safe callback to set the asyncio stop event.
        # - Input: None.
        # - Returns: None.
        def _request_stop() -> None:
            # Set the stop event so the async task can exit gracefully.
            if self._stop_event is not None:
                self._stop_event.set()

        # Ask the asyncio loop to execute the stop request in its own thread.
        loop.call_soon_threadsafe(_request_stop)

    # Summary:
    # - Return a copy of the most recent body names list.
    # - Input: `self`.
    # - Returns: Copy of body names (list[str]).
    def body_names_snapshot(self) -> list[str]:
        # Lock to keep the list consistent while copying.
        with self._lock:
            return list(self._body_names)

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
    # - Run the asyncio event loop for streaming in a background thread.
    # - Input: `self`, `ip_address` (str).
    # - Returns: None.
    def _run_stream_loop(self, ip_address: str) -> None:
        # Create a dedicated asyncio "event loop" for QTM streaming.
        #
        # Asyncio can be confusing at first, so the key idea is:
        # - The event loop is like a tiny scheduler that repeatedly:
        #   1) checks which async tasks are ready to continue,
        #   2) runs them a little bit,
        #   3) then waits (without busy-looping) for I/O or timers.
        #
        # IMPORTANT: event loops are thread-specific.
        # - Qt already owns the *main* thread because `app.exec()` runs the UI loop there.
        # - If we ran an asyncio loop in the main thread, the UI would freeze.
        # - So we start a *second* loop in this background thread dedicated to networking/QTM.
        #
        # `asyncio.new_event_loop()` creates a fresh loop object that is not attached to any thread yet.
        loop = asyncio.new_event_loop()
        # `asyncio.set_event_loop(loop)` registers this loop as "the current loop" for THIS thread.
        # This matters because many asyncio helpers (e.g., `asyncio.Event()`, `asyncio.get_event_loop()`)
        # assume there is a current loop and will bind themselves to it.
        asyncio.set_event_loop(loop)

        # Store the loop and stop event so the UI thread can signal a stop.
        with self._lock:
            self._loop = loop
            self._stop_event = asyncio.Event()

        try:
            # Run the async streaming coroutine until it exits or is stopped.
            loop.run_until_complete(self._stream_main(ip_address))
        finally:
            # Always close the loop to release resources when the thread exits.
            loop.close()
            with self._lock:
                self._loop = None
                self._stop_event = None

    # Summary:
    # - Async coroutine that connects to QTM and streams 6D residual data.
    # - Input: `self`, `ip_address` (str).
    # - Returns: None.
    async def _stream_main(self, ip_address: str) -> None:
        connection = None

        try:
            # Attempt to connect to QTM using the requested protocol version.
            connection = await qtm_rt.connect(
                ip_address,
                port=QTM_PORT,
                version=QTM_PROTOCOL_VERSION,
            )

            # Report a failure if the connection could not be established.
            if connection is None:
                self._safe_emit(
                    self._proxy.state_changed.emit,
                    False, f"Could not connect to QTM at {ip_address}:{QTM_PORT}"
                )
                return

            # If a stop was requested while connecting, close immediately.
            if self._stop_event is not None and self._stop_event.is_set():
                connection.disconnect()
                self._safe_emit(
                    self._proxy.state_changed.emit, False, "Stream stopped before connect."
                )
                return

            # Load body names before streaming so labels are ready for the first packet.
            body_names = await self._load_body_names(connection)
            with self._lock:
                self._body_names = list(body_names)

            # Notify the UI that streaming has started successfully.
            self._safe_emit(
                self._proxy.state_changed.emit,
                True, f"Connected to QTM at {ip_address}:{QTM_PORT}"
            )

            # Start streaming 6D residual frames and route packets to the callback.
            # The SDK returns a status string immediately, so we must keep the loop alive.
            stream_result = await connection.stream_frames(
                components=["6dres"],
                on_packet=self._on_packet,
            )
            # Normalize the SDK return (can be bytes) so comparisons are stable.
            if isinstance(stream_result, bytes):
                stream_result = stream_result.decode(errors="ignore")
            # If QTM did not accept the stream request, stop and notify the UI.
            if stream_result != "Ok":
                self._safe_emit(
                    self._proxy.state_changed.emit,
                    False, f"Stream request failed: {stream_result}"
                )
                return

            # Stay alive until the UI requests a stop or QTM drops the transport.
            if self._stop_event is not None:
                while not self._stop_event.is_set():
                    # If QTM drops the connection, exit so the UI can reset.
                    if not connection.has_transport():
                        self._safe_emit(
                            self._proxy.state_changed.emit,
                            False, "Connection closed by QTM."
                        )
                        return
                    # Sleep briefly to avoid busy-waiting on the asyncio loop.
                    await asyncio.sleep(0.1)

            # Ask QTM to stop streaming before disconnecting.
            if connection.has_transport():
                await connection.stream_frames_stop()

            # Inform the UI that streaming has stopped so it can reset controls.
            self._safe_emit(self._proxy.state_changed.emit, False, "Stream stopped.")
        except Exception as exc:
            # Report unexpected errors so the UI can recover and show feedback.
            self._safe_emit(self._proxy.state_changed.emit, False, f"Stream error: {exc}")
        finally:
            # Disconnect if the transport is still active.
            if connection is not None and connection.has_transport():
                connection.disconnect()

    # Summary:
    # - Handle incoming QTM packets and forward formatted text to the UI.
    # - Input: `self`, `packet` (qtm_rt.QRTPacket).
    # - Returns: None.
    def _on_packet(self, packet) -> None:
        # Read the 6D residual component once so text and plot stay in sync.
        result = packet.get_6d_residual()

        # Snapshot body names so formatting uses stable labels.
        with self._lock:
            body_names = list(self._body_names)

        # Format the packet defensively so missing components do not break the stream.
        text = self._format_6d_residual_text(packet, body_names, result)
        # Build a stable pose mapping aligned with body_names for the plot.
        poses_qtm = self._extract_6d_poses(result, body_names)
        # Timestamp the rigid body packet at PC-receive time for software coupling.
        rigidbody_ts_ms = _now_ms()
        # Emit the text from the worker thread to the UI thread.
        self._safe_emit(self._proxy.text_ready.emit, text)
        # Emit the timestamped rigid body packet from worker thread to UI thread.
        self._safe_emit(self._proxy.poses_ready.emit, rigidbody_ts_ms, poses_qtm)

        # Forward the result to the recorder without touching any UI state.
        self._recorder.handle_6d_residual_result(result)


# Summary:
# - Main mocap panel widget that renders the Designer UI and orchestrates the streaming lifecycle.
# - What it owns: UI state (enabled/disabled controls, button toggle text), plot timer, and worker instances.
# - What it ensures: the Qt UI never freezes (no blocking calls on the UI thread), and streaming/recording can
#   be started and stopped repeatedly without restarting the application.
class MocapWidget(QWidget):
    # Signal: emit one rigid body packet per mocap update for downstream coupling.
    sig_rigidbody_packet = Signal(int, object)

    # Summary:
    # - Initialize the UI and wire signal handlers for streaming control.
    # - Input: `self`.
    # - Returns: None.
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_Mocap()
        self.ui.setupUi(self)

        # Keep the text area readable but non-editable for incoming stream data.
        self.ui.plainTextEdit_mocap_textStream.setReadOnly(True)
        # Enable the widget so stream output is visible even though editing is blocked.
        self.ui.plainTextEdit_mocap_textStream.setEnabled(True)

        # Track the streaming state so the button can toggle cleanly.
        self._stream_state = "idle"

        # Build the Matplotlib canvas and place it in the prepared UI container.
        self._mocap_canvas = Mocap3DCanvas(self)
        self.ui.verticalLayout_mocap_matplotlib.addWidget(self._mocap_canvas)

        # Cache the latest QTM poses for the throttled plot updates.
        self._latest_poses_qtm: dict[str, Optional[np.ndarray]] = {}
        # Cache the latest rigid body packet fields for coupled stream consumers.
        self._latest_rigidbody_ts_ms: Optional[int] = None
        self._latest_rigidbody_data: Optional[object] = None

        # Create a timer that redraws the plot at a fixed rate on the UI thread.
        self._plot_timer = QTimer(self)
        self._plot_timer.setObjectName("mocapPlotTimer")
        self._plot_timer.setInterval(33)
        # Signal connection: update the plot at a fixed rate to avoid per-packet redraws.
        self._plot_timer.timeout.connect(self._on_mocapPlotTimer_timeout)

        # Create a signal proxy so thread callbacks can safely update the UI.
        self._stream_proxy = MocapStreamProxy()
        # Signal connection: route background text updates into the UI thread.
        self._stream_proxy.text_ready.connect(self._on_mocapStreamProxy_text_ready)
        # Signal connection: route background pose updates into the UI thread.
        self._stream_proxy.poses_ready.connect(self._on_mocapStreamProxy_poses_ready)
        # Signal connection: react to state changes from the stream thread (connected, stopped, errors).
        self._stream_proxy.state_changed.connect(self._on_mocapStreamProxy_state_changed)
        # Signal connection: show recording warnings or info on the UI thread.
        self._stream_proxy.record_message.connect(self._on_mocapStreamProxy_record_message)
        # Signal connection: stop recording safely from background threads.
        self._stream_proxy.record_stop.connect(self._on_mocapStreamProxy_record_stop)

        # Create worker instances that own the stream and recording threads.
        self._record_worker = CsvRecordWorker(self._stream_proxy)
        self._stream_worker = QtmStreamWorker(self._stream_proxy, self._record_worker)

        # Connect the open/close button to the toggle handler.
        self.ui.pushButton_mocap_openStream.clicked.connect(
            self._on_pushButton_mocap_openStream_clicked
        )
        # Signal connection: clear the record directory when the clear button is clicked.
        self.ui.pushButton_mocap_recorddirClear.clicked.connect(
            self._on_pushButton_mocap_recorddirClear_clicked
        )
        # Signal connection: open a directory picker for the record directory.
        self.ui.pushButton_mocap_recorddirBrowse.clicked.connect(
            self._on_pushButton_mocap_recorddirBrowse_clicked
        )
        # Signal connection: toggle recording when the record button is clicked.
        self.ui.pushButton_mocap_record.clicked.connect(
            self._on_pushButton_mocap_record_clicked
        )

        # Ensure the system combo starts on Qualisys, since that is the only supported option.
        self.ui.comboBox_mocap_systemSelect.setCurrentText("Qualisys")
        # Disable recording until streaming is active.
        self.ui.pushButton_mocap_record.setEnabled(False)

    # Summary:
    # - Slot function that handles the open/close stream button click.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_mocap_openStream_clicked(self) -> None:
        # Toggle based on current state so the user can start or stop streaming.
        if self._stream_state == "idle":
            self._start_stream()
        else:
            self._stop_stream()

    # Summary:
    # - Slot function that clears the record directory line edit.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_mocap_recorddirClear_clicked(self) -> None:
        # UI state change: clear the selected record directory text.
        self.ui.lineEdit_mocap_recorddir.setText("")

    # Summary:
    # - Slot function that opens a directory picker for the record directory.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_mocap_recorddirBrowse_clicked(self) -> None:
        # Use the current text as the starting folder when it exists.
        current_path = self.ui.lineEdit_mocap_recorddir.text().strip()
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
        self.ui.lineEdit_mocap_recorddir.setText(absolute_path)

    # Summary:
    # - Slot function that starts or stops CSV recording.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_mocap_record_clicked(self) -> None:
        # Toggle based on current recording state to keep the button behavior predictable.
        if self._record_worker.is_active():
            self._stop_recording("Recording stopped.")
            return
        self._start_recording()

    # Summary:
    # - Resolve the IP address from the line edit (or placeholder).
    # - Input: `self`.
    # - Returns: The IP address string (str), or an empty string if unavailable.
    def _resolve_ip_address(self) -> str:
        # Prefer the explicit user input, but fall back to the placeholder.
        ip_address = self.ui.lineEdit_mocap_ip.text().strip()
        if not ip_address:
            ip_address = self.ui.lineEdit_mocap_ip.placeholderText().strip()
        return ip_address

    # Summary:
    # - Validate that a string looks like a real IP address (IPv4 or IPv6).
    # - Input: `ip_address` (str).
    # - Returns: True when valid, otherwise False (bool).
    def _is_ip_address_valid(self, ip_address: str) -> bool:
        # Use the stdlib parser so we only accept well-formed IP addresses.
        try:
            ipaddress.ip_address(ip_address)
        except ValueError:
            return False
        return True

    # Summary:
    # - Start the Qualisys streaming thread and update the UI for a pending connection.
    # - Input: `self`.
    # - Returns: None.
    def _start_stream(self) -> None:
        # Validate the selected mocap system because only Qualisys is supported.
        system = self.ui.comboBox_mocap_systemSelect.currentText().strip()
        if system != "Qualisys":
            QMessageBox.warning(
                self,
                "Unsupported System",
                "Only the Qualisys system is supported right now.",
            )
            return

        # Resolve the target IP address before attempting a connection.
        ip_address = self._resolve_ip_address()
        if not ip_address:
            QMessageBox.warning(
                self,
                "Missing IP Address",
                "Please enter the Qualisys system IP address.",
            )
            return

        # Confirm the IP address looks valid so we fail fast on obvious typos.
        if not self._is_ip_address_valid(ip_address):
            QMessageBox.warning(
                self,
                "Invalid IP Address",
                "Please enter a valid IP address (for example, 192.168.0.10).",
            )
            return

        # Prevent duplicate start requests while we spin up the background loop.
        if self._stream_state != "idle":
            return

        # Mark the UI as busy so users know we are trying to connect.
        self._stream_state = "connecting"
        # Disable input controls while a connection is pending to avoid mid-connection changes.
        self.ui.comboBox_mocap_systemSelect.setEnabled(False)
        self.ui.lineEdit_mocap_ip.setEnabled(False)
        # UI state change: disable recording while connecting.
        self.ui.pushButton_mocap_record.setEnabled(False)
        # Change the button label so a second click stops the connection attempt.
        self.ui.pushButton_mocap_openStream.setText("Stop Stream")
        # Provide immediate feedback in the text area.
        self.ui.plainTextEdit_mocap_textStream.setPlainText(
            f"Connecting to {ip_address}:{QTM_PORT}..."
        )

        # Clear the plot and cache so stale points are not shown during a new connect.
        self._reset_plot_state()

        # Launch the asyncio event loop in a background thread so the UI stays responsive.
        self._stream_worker.start(ip_address)

    # Summary:
    # - Stop the Qualisys stream by signaling the asyncio loop to shut down.
    # - Input: `self`.
    # - Returns: None.
    def _stop_stream(self) -> None:
        # Stop plot updates immediately so the UI does not redraw after the user stops.
        self._plot_timer.stop()
        # Clear the plot and cache so the stopped state looks empty.
        self._reset_plot_state()

        # Signal the streaming worker to stop its asyncio loop safely.
        self._stream_worker.stop()

    # Summary:
    # - Slot function that updates the UI text area with the latest stream data.
    # - Input: `self`, `text` (str).
    # - Returns: None.
    def _on_mocapStreamProxy_text_ready(self, text: str) -> None:
        # Replace the text each time so only the latest frame is shown.
        self.ui.plainTextEdit_mocap_textStream.setPlainText(text)

    # Summary:
    # - Slot function that reacts to stream state changes and updates UI controls accordingly.
    # - Input: `self`, `is_running` (bool), `message` (str).
    # - Returns: None.
    def _on_mocapStreamProxy_state_changed(self, is_running: bool, message: str) -> None:
        # Always display the latest status message in the text area.
        if message:
            self.ui.plainTextEdit_mocap_textStream.setPlainText(message)

        if is_running:
            # Mark the UI as streaming so users know the connection is live.
            self._stream_state = "streaming"
            self.ui.pushButton_mocap_openStream.setText("Stop Stream")
            # Keep inputs disabled while streaming to prevent changing live settings.
            self.ui.comboBox_mocap_systemSelect.setEnabled(False)
            self.ui.lineEdit_mocap_ip.setEnabled(False)
            # Clear cached plot state and start the timer for throttled redraws.
            self._reset_plot_state()
            self._plot_timer.start()
            # UI state change: enable recording only when streaming is active.
            self.ui.pushButton_mocap_record.setEnabled(True)
            return

        # Reset UI controls after the stream stops or fails.
        self._stream_state = "idle"
        self.ui.pushButton_mocap_openStream.setText("Open Stream")
        # Re-enable inputs so the user can change settings before reconnecting.
        self.ui.comboBox_mocap_systemSelect.setEnabled(True)
        self.ui.lineEdit_mocap_ip.setEnabled(True)
        # Stop plot updates immediately when the stream stops or drops.
        self._plot_timer.stop()
        # Clear plot data so the stopped state is visually empty.
        self._reset_plot_state()
        # Stop recording if the stream is no longer running.
        if self._record_worker.is_active():
            self._stop_recording("Recording stopped because streaming ended.")
        # UI state change: disable recording while idle.
        self.ui.pushButton_mocap_record.setEnabled(False)

    # Summary:
    # - Slot function that displays recording-related messages on the UI thread.
    # - Input: `self`, `message` (str).
    # - Returns: None.
    def _on_mocapStreamProxy_record_message(self, message: str) -> None:
        # Show the latest recording message so the user can react quickly.
        if message:
            self.ui.plainTextEdit_mocap_textStream.setPlainText(message)

    # Summary:
    # - Slot function that stops recording when background threads request it.
    # - Input: `self`, `reason` (str).
    # - Returns: None.
    def _on_mocapStreamProxy_record_stop(self, reason: str) -> None:
        # Stop recording safely on the UI thread when an error or overload occurs.
        if self._record_worker.is_active():
            self._stop_recording(reason)
        elif reason:
            # Still surface the message even if recording already stopped.
            self.ui.plainTextEdit_mocap_textStream.setPlainText(reason)

    # Summary:
    # - Slot function that caches the latest QTM poses for the plot timer.
    # - Input: `self`, `rigidbody_ts_ms` (int), `rigidbody_data` (dict[str, np.ndarray | None]).
    # - Returns: None.
    def _on_mocapStreamProxy_poses_ready(
        self, rigidbody_ts_ms: int, rigidbody_data: dict[str, Optional[np.ndarray]]
    ) -> None:
        # Cache poses so the UI timer can update the plot at a fixed rate.
        self._latest_poses_qtm = rigidbody_data
        # Cache the latest rigid body packet fields for pull-style coupling usage.
        self._latest_rigidbody_ts_ms = int(rigidbody_ts_ms)
        self._latest_rigidbody_data = rigidbody_data
        # Emit the packet outward so the coupled controller can match by timestamp.
        self.sig_rigidbody_packet.emit(
            self._latest_rigidbody_ts_ms, self._latest_rigidbody_data
        )

    # Summary:
    # - Slot function that redraws the plot on a fixed timer interval.
    # - Input: `self`.
    # - Returns: None.
    def _on_mocapPlotTimer_timeout(self) -> None:
        # Update the plot on the UI thread using the cached poses.
        self._mocap_canvas.update_poses(self._latest_poses_qtm)

    # Summary:
    # - Clear cached poses and reset the matplotlib scatter to empty.
    # - Input: `self`.
    # - Returns: None.
    def _reset_plot_state(self) -> None:
        # Clear cached poses so the timer does not redraw stale data.
        self._latest_poses_qtm = {}
        self._latest_rigidbody_ts_ms = None
        self._latest_rigidbody_data = None
        # Clear the scatter so the user sees an empty plot when stopped.
        self._mocap_canvas.clear_points()

    # Summary:
    # - Build the CSV header for a frozen list of rigid body names.
    # - Input: `body_names` (list[str]).
    # - Returns: CSV header row (list[str]).
    @staticmethod
    def _build_record_header(body_names: list[str]) -> list[str]:
        # Start with the timestamp column required by the schema.
        header = ["utc_epoch_ms"]
        for name in body_names:
            # Append the quaternion columns first (w, x, y, z), then translation.
            header.extend(
                [
                    f"{name}_q1",
                    f"{name}_q2",
                    f"{name}_q3",
                    f"{name}_q4",
                    f"{name}_t1",
                    f"{name}_t2",
                    f"{name}_t3",
                ]
            )
        return header

    # Summary:
    # - Build a unique CSV file path for a new recording session.
    # - Input: `record_dir` (str).
    # - Returns: Full file path for the CSV (str).
    @staticmethod
    def _build_record_file_path(record_dir: str) -> str:
        # Use UTC in the filename so sessions are sortable and unambiguous.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        base_name = f"qtm_{timestamp}.csv"
        file_path = os.path.join(record_dir, base_name)

        # Add a numeric suffix when a file with the same name already exists.
        if not os.path.exists(file_path):
            return file_path

        for suffix in range(1, 1000):
            candidate = os.path.join(record_dir, f"qtm_{timestamp}_{suffix:03d}.csv")
            if not os.path.exists(candidate):
                return candidate

        # Fall back to the base name if we somehow exhausted all suffixes.
        return file_path

    # Summary:
    # - Start a new CSV recording session on the UI thread.
    # - Input: `self`.
    # - Returns: None.
    def _start_recording(self) -> None:
        # Require an active stream so recording stays aligned to live packets.
        if self._stream_state != "streaming":
            QMessageBox.warning(
                self,
                "Stream Not Active",
                "Start streaming before recording.",
            )
            return

        # Validate the record directory so we fail fast on missing paths.
        record_dir = self.ui.lineEdit_mocap_recorddir.text().strip()
        if not record_dir:
            QMessageBox.warning(
                self,
                "Missing Record Directory",
                "Please choose a record directory before recording.",
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

        # Freeze the body names at record start so the CSV schema is stable.
        record_body_names = self._stream_worker.body_names_snapshot()
        if not record_body_names:
            # Fall back to the latest pose keys when QTM parameters are unavailable.
            if not self._latest_poses_qtm:
                QMessageBox.warning(
                    self,
                    "Body Names Missing",
                    "Rigid body names are not available yet. Try again once streaming is active.",
                )
                return
            record_body_names = list(self._latest_poses_qtm.keys())

        # Build the output path and header before starting the writer thread.
        record_file_path = self._build_record_file_path(record_dir)
        record_header = self._build_record_header(record_body_names)

        # Start the writer thread so file I/O never blocks the stream thread.
        self._record_worker.start(record_file_path, record_header, record_body_names)

        # UI state change: lock the record directory while recording is active.
        self.ui.lineEdit_mocap_recorddir.setEnabled(False)
        self.ui.pushButton_mocap_recorddirBrowse.setEnabled(False)
        self.ui.pushButton_mocap_recorddirClear.setEnabled(False)
        # UI state change: update the record button to show stop intent.
        self.ui.pushButton_mocap_record.setText("Stop Recording")
        # Provide immediate feedback about the active recording file.
        self.ui.plainTextEdit_mocap_textStream.setPlainText(
            f"Recording to {record_file_path}"
        )

    # Summary:
    # - Stop the active CSV recording session and release writer resources.
    # - Input: `self`, `reason` (str).
    # - Returns: None.
    def _stop_recording(self, reason: str = "") -> None:
        # Avoid redundant work when recording is already stopped.
        if not self._record_worker.is_active():
            return

        # Stop the recording worker so it can flush and close cleanly.
        self._record_worker.stop(reason)

        # UI state change: unlock the record directory controls.
        self.ui.lineEdit_mocap_recorddir.setEnabled(True)
        self.ui.pushButton_mocap_recorddirBrowse.setEnabled(True)
        self.ui.pushButton_mocap_recorddirClear.setEnabled(True)
        # UI state change: restore the record button text.
        self.ui.pushButton_mocap_record.setText("Record")
        # Keep the record button enabled only while streaming.
        self.ui.pushButton_mocap_record.setEnabled(self._stream_state == "streaming")

        # Show the stop reason so the user understands why recording ended.
        if reason:
            self.ui.plainTextEdit_mocap_textStream.setPlainText(reason)

    # Summary:
    # - Ensure the streaming thread is stopped before the widget closes.
    # - Input: `self`, `event` (Qt close event).
    # - Returns: None.
    def closeEvent(self, event) -> None:
        # Stop recording so the writer thread can close the file cleanly.
        if self._record_worker.is_active():
            self._stop_recording("Recording stopped because the window closed.")
        # Stop streaming so QTM connections are released when the window closes.
        self._stop_stream()
        # Stop the plot timer to avoid redraws after the widget is closed.
        self._plot_timer.stop()
        event.accept()


# Summary:
# - Application entry point that shows the MocapWidget window.
# - Input: None (reads command-line args via sys.argv).
# - Returns: None.
def main() -> None:
    # Import sys here to keep module imports focused for the widget.
    import sys

    # Create and run the Qt application.
    app = QApplication(sys.argv)
    window = MocapWidget()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
