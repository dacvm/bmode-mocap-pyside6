"""PySide6 widget that wires up the Volume UI file selection controls."""

# Standard library helpers for path normalization and threading.
import os
import tempfile
import threading
from datetime import datetime
from typing import Optional

# Third-party imports for plotting and numeric helpers.
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QProcess, Qt, Signal, QTimer
from PySide6.QtGui import QTextCursor

# Qt widgets for the main window and file dialogs.
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QMessageBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Generated UI class from Qt Designer (do not edit the _ui.py file).
from volume_ui import Ui_Form as Ui_Volume

from helpers.mha_reader import MhaReader
from helpers.mha_volume import MhaVolume
from helpers.mha_volume_geometry_builder import (
    DEFAULT_MAX_POINTS,
    DEFAULT_THRESHOLD,
    GeometryBuildingResult,
    MhaVolumeGeometryBuilder,
)

DEFAULT_CONFIG_FILE = (
    "configs/PlusDeviceSet_fCal_Epiphan_NDIPolaris_RadboudUMC_20241219_150400.xml"
)
DEFAULT_SEQUENCE_FILE = "seqs/SequenceRecording_2024-12-20_14-47-41.mha"
DEFAULT_OUTPUT_DIR = "seqs/"
DEFAULT_VOLUME_FILE = "seqs/VolumeOutput_2024-12-20_14-48-05.mha"
VOLUME_THRESHOLD_DEBOUNCE_MS = 100
DEFAULT_VOLUME_RECONSTRUCTOR_EXE = "C:/Users/DennisChristie/PlusApp-2.9.0.20240320-Win64/bin/VolumeReconstructor.exe"
VOLUME_RECONSTRUCTOR_EXE_ENV = "PLUS_VOLUME_RECONSTRUCTOR_EXE"
RECON_OUTPUT_BUFFER_LIMIT = 8000
RECON_LOG_TEXT_LIMIT = 25000


# Summary:
# - Matplotlib canvas that renders volume geometry points in 3D.
# - What it does: owns a Figure, a 3D Axes, and a persistent scatter handle for updates.
class Volume3DCanvas(FigureCanvas):
    # Summary:
    # - Initialize the 3D scatter plot with an empty data set.
    # - Input: `self`, `parent` (QWidget | None).
    # - Returns: None.
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        # Build the figure and axes once so updates are just data changes.
        figure = Figure()
        axes = figure.add_subplot(111, projection="3d")
        super().__init__(figure)

        # Keep a reference to the axes for fast updates later.
        self._axes = axes
        # Create the scatter handle once so updates only move existing points.
        self._scatter = self._axes.scatter([], [], [], s=2)

        # Disable autoscale so the camera stays stable after we set limits.
        self._axes.set_autoscale_on(False)
        # Label axes so users know the coordinate directions.
        self._axes.set_xlabel("X")
        self._axes.set_ylabel("Y")
        self._axes.set_zlabel("Z")
        self._axes.set_proj_type("ortho")

        # Track whether bounds have been set at least once.
        self._has_bounds = False

        # Ensure the canvas is parented to the widget tree if provided.
        if parent is not None:
            self.setParent(parent)

    # Summary:
    # - Update the scatter points and axes bounds from a geometry-building result.
    # - Input: `self`, `points_xyz` (np.ndarray), `intensities` (np.ndarray | None),
    #   `min_xyz` (np.ndarray | None), `max_xyz` (np.ndarray | None).
    # - Returns: None.
    def update_points(
        self,
        points_xyz: np.ndarray,
        intensities: Optional[np.ndarray],
        min_xyz: Optional[np.ndarray],
        max_xyz: Optional[np.ndarray],
    ) -> None:
        # Ignore intensities for now; the plot uses a single color.
        _ = intensities

        # Clear the plot when there are no points to show.
        if points_xyz is None or points_xyz.size == 0:
            self.clear_points()
            return

        # Update the existing scatter in place to avoid reallocating artists.
        self._scatter._offsets3d = (
            points_xyz[:, 0],
            points_xyz[:, 1],
            points_xyz[:, 2],
        )

        # Update bounds when they are available so the view fits the data.
        if min_xyz is not None and max_xyz is not None:
            span = max_xyz - min_xyz
            margin = np.maximum(span * 0.1, 1.0)

            self._axes.set_xlim(min_xyz[0] - margin[0], max_xyz[0] + margin[0])
            self._axes.set_ylim(min_xyz[1] - margin[1], max_xyz[1] + margin[1])
            self._axes.set_zlim(min_xyz[2] - margin[2], max_xyz[2] + margin[2])
            self._has_bounds = True
        elif not self._has_bounds:
            # Keep a fallback view if bounds are missing on the first update.
            self._axes.set_xlim(0.0, 1.0)
            self._axes.set_ylim(0.0, 1.0)
            self._axes.set_zlim(0.0, 1.0)

        # Request a redraw without blocking the UI thread.
        self.draw_idle()

    # Summary:
    # - Clear the scatter points and reset bounds tracking.
    # - Input: `self`.
    # - Returns: None.
    def clear_points(self) -> None:
        # Empty the scatter data so the plot visibly clears.
        self._scatter._offsets3d = ([], [], [])
        # Reset bounds tracking so we autoscale on the next valid result.
        self._has_bounds = False
        # Schedule a redraw so the clear is visible to the user.
        self.draw_idle()


# Summary:
# - Qt signal bridge that delivers geometry-building results to the UI thread.
# - What it does: exposes a result_ready signal that carries (result, request_id).
class VolumeBuildGeometryProxy(QObject):
    result_ready = Signal(object, int)

    # Summary:
    # - Initialize the proxy and assign a stable objectName for slot naming.
    # - Input: `self`.
    # - Returns: None.
    def __init__(self) -> None:
        super().__init__()
        # Give the proxy a name so slot handlers can follow the naming convention.
        self.setObjectName("volumeBuildGeometryProxy")


# Summary:
# - Background worker that runs volume geometry building without blocking the UI thread.
# - What it does: starts a thread per request and emits results through a proxy signal.
class VolumeBuildGeometryWorker:
    # Summary:
    # - Initialize the worker with a signal proxy and geometry builder.
    # - Input: `self`, `proxy` (VolumeBuildGeometryProxy), `geometry_builder`
    #   (MhaVolumeGeometryBuilder).
    # - Returns: None.
    def __init__(
        self,
        proxy: VolumeBuildGeometryProxy,
        geometry_builder: MhaVolumeGeometryBuilder,
    ) -> None:
        # Store references so background threads can compute and emit results.
        self._proxy = proxy
        self._geometry_builder = geometry_builder

    # Summary:
    # - Start a geometry build in a background thread.
    # - Input: `self`, `threshold` (float), `max_points` (int), `request_id` (int).
    # - Returns: None.
    def start(self, threshold: float, max_points: int, request_id: int) -> None:
        # Thread start: run geometry building in the background so the UI stays responsive.
        thread = threading.Thread(
            target=self._run,
            args=(threshold, max_points, request_id),
            daemon=True,
        )
        thread.start()

    # Summary:
    # - Thread worker that builds geometry and emits results.
    # - Input: `self`, `threshold` (float), `max_points` (int), `request_id` (int).
    # - Returns: None.
    def _run(self, threshold: float, max_points: int, request_id: int) -> None:
        # Compute the geometry build off the UI thread.
        result = self._geometry_builder.build_geometry(
            threshold=threshold,
            max_points=max_points,
            mode="preview",
        )
        # Emit results safely back to the UI thread.
        self._safe_emit(self._proxy.result_ready.emit, result, request_id)

    # Summary:
    # - Emit a proxy signal safely when the UI may have already been destroyed.
    # - Input: `self`, `emit_fn` (callable), `args` (tuple[object, ...]).
    # - Returns: None.
    def _safe_emit(self, emit_fn, *args) -> None:
        # Avoid crashing if Qt deletes the proxy while background threads still run.
        try:
            emit_fn(*args)
        except RuntimeError:
            return


# Summary:
# - Main widget that binds the Volume UI controls to basic file browsing behavior.
# - What it does: Connects clear/browse buttons to line edits and sets default paths.
class VolumeWidget(QWidget):
    # Summary:
    # - Initialize the widget, build the UI, and connect signals.
    # - Input: `self`.
    # - Returns: None.
    def __init__(self) -> None:
        super().__init__()
        # Build the Qt Designer UI into this widget.
        self.ui = Ui_Volume()
        self.ui.setupUi(self)

        # # UI state change: preload default paths so users see the expected files.
        # self.ui.lineEdit_volume_configfile.setText(DEFAULT_CONFIG_FILE)
        # self.ui.lineEdit_volume_seqfile.setText(DEFAULT_SEQUENCE_FILE)
        # self.ui.lineEdit_volume_outputdir.setText(DEFAULT_OUTPUT_DIR)
        # self.ui.lineEdit_volume_volfile.setText(DEFAULT_VOLUME_FILE)

        # Signal connection: clear the config file line edit.
        self.ui.pushButton_volume_configfileClear.clicked.connect(
            self._on_pushButton_volume_configfileClear_clicked
        )
        # Signal connection: browse for a config XML file.
        self.ui.pushButton_volume_configfileBrowse.clicked.connect(
            self._on_pushButton_volume_configfileBrowse_clicked
        )
        # Signal connection: clear the sequence file line edit.
        self.ui.pushButton_volume_seqfileClear.clicked.connect(
            self._on_pushButton_volume_seqfileClear_clicked
        )
        # Signal connection: browse for a sequence MHA file.
        self.ui.pushButton_volume_seqfileBrowse.clicked.connect(
            self._on_pushButton_volume_seqfileBrowse_clicked
        )
        # Signal connection: clear the output directory line edit.
        self.ui.pushButton_volume_outputdirClear.clicked.connect(
            self._on_pushButton_volume_outputdirClear_clicked
        )
        # Signal connection: browse for an output directory.
        self.ui.pushButton_volume_outputdirBrowse.clicked.connect(
            self._on_pushButton_volume_outputdirBrowse_clicked
        )
        # Signal connection: browse for a volume MHA file.
        self.ui.pushButton_volume_volfileBrowse.clicked.connect(
            self._on_pushButton_volume_volfileBrowse_clicked
        )
        # Signal connection: load the selected volume MHA file.
        self.ui.pushButton_volume_volload.clicked.connect(
            self._on_pushButton_volume_volload_clicked
        )
        # Signal connection: start or stop volume reconstruction.
        self.ui.pushButton_volume_reconstruct.clicked.connect(
            self._on_pushButton_volume_reconstruct_clicked
        )

        # UI state change: set a default threshold range/value so the slider is usable.
        self.ui.horizontalSlider_volume_threshold.setMinimum(0)
        self.ui.horizontalSlider_volume_threshold.setMaximum(255)
        self.ui.horizontalSlider_volume_threshold.setValue(DEFAULT_THRESHOLD)

        # Build the 3D scatter canvas and add it to the layout.
        self._volume_canvas = Volume3DCanvas(self.ui.widget_volume_scatter)
        self.ui.verticalLayout_volume_scatter.addWidget(self._volume_canvas)

        # Create a proxy to deliver geometry-building results back to the UI thread.
        self._build_geometry_proxy = VolumeBuildGeometryProxy()
        # Signal connection: apply geometry-building results on the UI thread.
        self._build_geometry_proxy.result_ready.connect(
            self._on_volumeBuildGeometryProxy_result_ready
        )

        # Create a debounce timer so slider drags do not spam heavy work.
        self._volume_threshold_timer = QTimer(self)
        self._volume_threshold_timer.setObjectName("volumeThresholdDebounceTimer")
        self._volume_threshold_timer.setSingleShot(True)
        self._volume_threshold_timer.setInterval(VOLUME_THRESHOLD_DEBOUNCE_MS)
        # Signal connection: fire geometry building once the user pauses slider movement.
        self._volume_threshold_timer.timeout.connect(
            self._on_volumeThresholdDebounceTimer_timeout
        )
        # Signal connection: debounce threshold changes from the slider.
        self.ui.horizontalSlider_volume_threshold.valueChanged.connect(
            self._on_horizontalSlider_volume_threshold_value_changed
        )

        # Keep a reader instance so we can reuse it for multiple loads.
        self._mha_reader = MhaReader()
        # Track the last loaded volume so other actions can reuse it later.
        self._mha_volume: Optional[MhaVolume] = None

        # Track the geometry builder and worker so we can rebuild when a new volume loads.
        self._geometry_builder: Optional[MhaVolumeGeometryBuilder] = None
        self._build_geometry_worker: Optional[VolumeBuildGeometryWorker] = None
        # Track pending slider values for the debounce timer.
        self._pending_threshold = DEFAULT_THRESHOLD
        # Track the maximum number of points to render for performance.
        self._max_points = DEFAULT_MAX_POINTS
        # Generation counter so stale worker results are ignored.
        self._build_geometry_request_id = 0

        # Track the external reconstruction process state and diagnostics.
        self._recon_process: Optional[QProcess] = None
        self._recon_state = "idle"
        self._recon_output_path = ""
        self._recon_stdout = ""
        self._recon_stderr = ""
        # Track the temporary reconstruction log dialog and its line buffers.
        self._recon_log_dialog: Optional[QDialog] = None
        self._recon_log_text: Optional[QTextEdit] = None
        self._recon_stdout_line_buffer = ""
        self._recon_stderr_line_buffer = ""

    # Summary:
    # - Resolve a safe starting path for file dialogs.
    # - What it does: Uses the line edit value when present, otherwise a default path.
    # - Input: `current_text` (str), `default_path` (str).
    # - Returns: Absolute path string to use as a dialog start location (str).
    def _resolve_start_path(self, current_text: str, default_path: str) -> str:
        # Prefer the user-provided path so dialogs reopen where they last browsed.
        base_path = current_text.strip() if current_text.strip() else default_path
        # Normalize to an absolute path so Qt gets a stable starting location.
        return os.path.abspath(base_path)

    # Summary:
    # - Slot function that clears the config file line edit.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_volume_configfileClear_clicked(self) -> None:
        # UI state change: clear the config file path.
        self.ui.lineEdit_volume_configfile.setText("")

    # Summary:
    # - Slot function that opens a config XML file picker.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_volume_configfileBrowse_clicked(self) -> None:
        # Use the current text or default file as the starting point.
        start_path = self._resolve_start_path(
            self.ui.lineEdit_volume_configfile.text(),
            DEFAULT_CONFIG_FILE,
        )
        # Open a file dialog filtered to XML files.
        selected_file, _filter = QFileDialog.getOpenFileName(
            self,
            "Select Configuration File",
            start_path,
            "XML Files (*.xml);;All Files (*)",
        )
        # Do nothing if the user cancels the dialog.
        if not selected_file:
            return
        # UI state change: store the selected file path.
        self.ui.lineEdit_volume_configfile.setText(os.path.abspath(selected_file))

    # Summary:
    # - Slot function that clears the sequence file line edit.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_volume_seqfileClear_clicked(self) -> None:
        # UI state change: clear the sequence file path.
        self.ui.lineEdit_volume_seqfile.setText("")

    # Summary:
    # - Slot function that opens a sequence MHA file picker.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_volume_seqfileBrowse_clicked(self) -> None:
        # Use the current text or default file as the starting point.
        start_path = self._resolve_start_path(
            self.ui.lineEdit_volume_seqfile.text(),
            DEFAULT_SEQUENCE_FILE,
        )
        # Open a file dialog filtered to MHA files.
        selected_file, _filter = QFileDialog.getOpenFileName(
            self,
            "Select Sequence File",
            start_path,
            "MHA Files (*.mha);;All Files (*)",
        )
        # Do nothing if the user cancels the dialog.
        if not selected_file:
            return
        # UI state change: store the selected file path.
        self.ui.lineEdit_volume_seqfile.setText(os.path.abspath(selected_file))

    # Summary:
    # - Slot function that clears the output directory line edit.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_volume_outputdirClear_clicked(self) -> None:
        # UI state change: clear the output directory path.
        self.ui.lineEdit_volume_outputdir.setText("")

    # Summary:
    # - Slot function that opens a directory picker for outputs.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_volume_outputdirBrowse_clicked(self) -> None:
        # Use the current text or default directory as the starting point.
        start_path = self._resolve_start_path(
            self.ui.lineEdit_volume_outputdir.text(),
            DEFAULT_OUTPUT_DIR,
        )
        # Open a directory-only picker so the user can select a folder.
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            start_path,
            QFileDialog.ShowDirsOnly,
        )
        # Do nothing if the user cancels the dialog.
        if not selected_dir:
            return
        # UI state change: store the selected directory path.
        self.ui.lineEdit_volume_outputdir.setText(os.path.abspath(selected_dir))

    # Summary:
    # - Slot function that opens a volume MHA file picker.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_volume_volfileBrowse_clicked(self) -> None:
        # Use the current text or default file as the starting point.
        start_path = self._resolve_start_path(
            self.ui.lineEdit_volume_volfile.text(),
            DEFAULT_VOLUME_FILE,
        )
        # Open a file dialog filtered to MHA files.
        selected_file, _filter = QFileDialog.getOpenFileName(
            self,
            "Select Volume File",
            start_path,
            "MHA Files (*.mha);;All Files (*)",
        )
        # Do nothing if the user cancels the dialog.
        if not selected_file:
            return
        # UI state change: store the selected file path.
        self.ui.lineEdit_volume_volfile.setText(os.path.abspath(selected_file))

    # Summary:
    # - Load a volume file, rebuild preview geometry, and update the plot.
    # - Input: `self`, `volfile_path` (str).
    # - Returns: True when the volume loads successfully; otherwise False.
    def _load_and_display_volume(self, volfile_path: str) -> bool:
        # Validate that a path was provided before we touch the filesystem.
        if not volfile_path.strip():
            # UI state change: notify the user the path is required.
            QMessageBox.warning(self, "Missing Volume File", "Select a volume .mha file first.")
            return False

        normalized_path = os.path.abspath(volfile_path)
        if not os.path.isfile(normalized_path):
            # UI state change: notify the user when the file is not found.
            QMessageBox.warning(
                self,
                "Volume File Not Found",
                f"The file does not exist:\n{normalized_path}",
            )
            return False

        # Load the .mha file into a MhaVolume container (no geometry build yet).
        try:
            self._mha_volume = self._mha_reader.read(normalized_path, use_memmap=True)
        except ValueError as exc:
            # UI state change: surface parsing errors without crashing the UI.
            QMessageBox.warning(self, "Volume Load Failed", str(exc))
            return False

        # Clear any existing plot data so the new volume starts fresh.
        self._volume_canvas.clear_points()

        # Configure the slider range based on the loaded volume dtype.
        self._configure_threshold_slider_for_volume(self._mha_volume)

        # Build a geometry builder and worker tied to this volume.
        self._geometry_builder = MhaVolumeGeometryBuilder(self._mha_volume)
        self._build_geometry_worker = VolumeBuildGeometryWorker(
            self._build_geometry_proxy,
            self._geometry_builder,
        )

        # Kick off an initial geometry build using the current slider value.
        self._pending_threshold = int(self.ui.horizontalSlider_volume_threshold.value())
        self._start_build_geometry(self._pending_threshold)
        return True

    # Summary:
    # - Slot function that loads the selected volume MHA file.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_volume_volload_clicked(self) -> None:
        # Read the volume file path from the UI and hand off to the shared loader.
        volfile_text = self.ui.lineEdit_volume_volfile.text().strip()
        # UI action: reuse the shared loader so validation and plotting stay consistent.
        self._load_and_display_volume(volfile_text)

    # Summary:
    # - Configure the threshold slider range based on the loaded volume dtype.
    # - Input: `self`, `mha_volume` (MhaVolume).
    # - Returns: None.
    def _configure_threshold_slider_for_volume(self, mha_volume: MhaVolume) -> None:
        # Guard against missing volume data so we do not crash on invalid loads.
        if mha_volume is None or mha_volume.data is None:
            return

        # Default to a safe range when the dtype is not an integer type.
        min_value = 0
        max_value = 1000
        dtype = mha_volume.data.dtype

        # Use dtype limits for integer volumes so the slider matches data range.
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            min_value = int(info.min)
            max_value = int(info.max)

        # Clamp the default threshold to the new range.
        default_value = DEFAULT_THRESHOLD
        if default_value < min_value:
            default_value = min_value
        if default_value > max_value:
            default_value = max_value

        # UI state change: update the slider range to match the volume.
        self.ui.horizontalSlider_volume_threshold.setMinimum(min_value)
        self.ui.horizontalSlider_volume_threshold.setMaximum(max_value)

        current_value = self.ui.horizontalSlider_volume_threshold.value()
        if current_value < min_value or current_value > max_value:
            # UI state change: reset the slider value when the old value is out of range.
            self.ui.horizontalSlider_volume_threshold.setValue(default_value)
            # Keep pending threshold aligned with the UI value.
            self._pending_threshold = int(default_value)

    # Summary:
    # - Start a background geometry build request with a new generation id.
    # - Input: `self`, `threshold` (int).
    # - Returns: None.
    def _start_build_geometry(self, threshold: int) -> None:
        # Do not start work until a volume and worker are ready.
        if self._geometry_builder is None or self._build_geometry_worker is None:
            return

        # Increment the generation counter so stale results are ignored.
        self._build_geometry_request_id += 1
        request_id = self._build_geometry_request_id

        # Thread start: run geometry building off the UI thread.
        self._build_geometry_worker.start(
            float(threshold),
            self._max_points,
            request_id,
        )

    # Summary:
    # - Slot function that debounces threshold slider changes.
    # - Input: `self`, `value` (int).
    # - Returns: None.
    def _on_horizontalSlider_volume_threshold_value_changed(self, value: int) -> None:
        # Cache the latest threshold so the timer can use it.
        self._pending_threshold = int(value)
        # Restart the single-shot timer to wait for the user to pause.
        self._volume_threshold_timer.start()

    # Summary:
    # - Slot function fired when the debounce timer expires.
    # - Input: `self`.
    # - Returns: None.
    def _on_volumeThresholdDebounceTimer_timeout(self) -> None:
        # Avoid starting work before a volume has been loaded.
        if self._mha_volume is None:
            return
        # Start a geometry build using the most recent slider value.
        self._start_build_geometry(self._pending_threshold)

    # Summary:
    # - Slot function that applies geometry-building results to the scatter plot.
    # - Input: `self`, `result` (GeometryBuildingResult), `request_id` (int).
    # - Returns: None.
    def _on_volumeBuildGeometryProxy_result_ready(
        self,
        result: GeometryBuildingResult,
        request_id: int,
    ) -> None:
        # Ignore stale results when a newer request has been issued.
        if request_id != self._build_geometry_request_id:
            return

        # Update the scatter plot with the latest geometry points.
        self._volume_canvas.update_points(
            result.points_xyz,
            result.intensities,
            result.min_xyz,
            result.max_xyz,
        )

    # Summary:
    # - Enable/disable reconstruction-related inputs while the external process runs.
    # - Input: `self`, `running` (bool).
    # - Returns: None.
    def _set_recon_ui_running(self, running: bool) -> None:
        # UI state change: freeze config inputs so the running process stays consistent.
        self.ui.lineEdit_volume_configfile.setEnabled(not running)
        self.ui.pushButton_volume_configfileClear.setEnabled(not running)
        self.ui.pushButton_volume_configfileBrowse.setEnabled(not running)

        # UI state change: freeze sequence inputs so the running process stays consistent.
        self.ui.lineEdit_volume_seqfile.setEnabled(not running)
        self.ui.pushButton_volume_seqfileClear.setEnabled(not running)
        self.ui.pushButton_volume_seqfileBrowse.setEnabled(not running)

        # UI state change: freeze output inputs so the running process keeps its target.
        self.ui.lineEdit_volume_outputdir.setEnabled(not running)
        self.ui.pushButton_volume_outputdirClear.setEnabled(not running)
        self.ui.pushButton_volume_outputdirBrowse.setEnabled(not running)

        # UI state change: prevent loading another volume while reconstruction runs.
        self.ui.pushButton_volume_volload.setEnabled(not running)
        self.ui.pushButton_volume_volfileBrowse.setEnabled(not running)

        # UI state change: update the reconstruct button text to indicate running state.
        if running:
            self.ui.pushButton_volume_reconstruct.setText("Reconstructing... (Stop)")
        else:
            self.ui.pushButton_volume_reconstruct.setText("Reconstruct Volume")

        # UI state change: show a busy cursor to communicate background work.
        if running:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    # Summary:
    # - Validate reconstruction inputs and normalize their paths.
    # - Input: `self`.
    # - Returns: tuple of (config_path, seq_path, out_dir) on success, otherwise None.
    def _validate_recon_inputs(self) -> Optional[tuple[str, str, str]]:
        # Read and trim the text first so we can detect missing inputs early.
        config_text = self.ui.lineEdit_volume_configfile.text().strip()
        seq_text = self.ui.lineEdit_volume_seqfile.text().strip()
        out_text = self.ui.lineEdit_volume_outputdir.text().strip()

        # Validation: ensure all required fields are filled in.
        if not config_text:
            QMessageBox.warning(self, "Missing Config File", "Select a config .xml file first.")
            return None
        if not seq_text:
            QMessageBox.warning(self, "Missing Sequence File", "Select a sequence .mha file first.")
            return None
        if not out_text:
            QMessageBox.warning(self, "Missing Output Folder", "Select an output folder first.")
            return None

        # Normalize to absolute paths for stable downstream behavior.
        config_path = os.path.abspath(config_text)
        seq_path = os.path.abspath(seq_text)
        out_dir = os.path.abspath(out_text)

        # Validation: config file must exist and look like an XML file.
        if not os.path.isfile(config_path):
            QMessageBox.warning(
                self,
                "Config File Not Found",
                f"The file does not exist:\n{config_path}",
            )
            return None
        if not config_path.lower().endswith(".xml"):
            QMessageBox.warning(
                self,
                "Invalid Config File",
                "The config file should end with .xml.",
            )
            return None

        # Validation: sequence file must exist and look like an MHA file.
        if not os.path.isfile(seq_path):
            QMessageBox.warning(
                self,
                "Sequence File Not Found",
                f"The file does not exist:\n{seq_path}",
            )
            return None
        if not seq_path.lower().endswith(".mha"):
            QMessageBox.warning(
                self,
                "Invalid Sequence File",
                "The sequence file should end with .mha.",
            )
            return None

        # Validation: output directory must be a folder we can write into.
        if os.path.exists(out_dir) and not os.path.isdir(out_dir):
            QMessageBox.warning(
                self,
                "Invalid Output Folder",
                f"The path is not a directory:\n{out_dir}",
            )
            return None

        # Create the output folder if it does not exist yet.
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Output Folder Error",
                f"Failed to create output folder:\n{out_dir}\n\n{exc}",
            )
            return None

        # Validation: verify we can write into the output folder.
        try:
            with tempfile.NamedTemporaryFile(
                dir=out_dir,
                prefix=".__volume_write_test_",
                delete=True,
            ) as temp_file:
                temp_file.write(b"")
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Output Folder Not Writable",
                f"Cannot write to output folder:\n{out_dir}\n\n{exc}",
            )
            return None

        return config_path, seq_path, out_dir

    # Summary:
    # - Build a unique output volume file path inside the output folder.
    # - Input: `self`, `out_dir` (str).
    # - Returns: Unique output file path (str).
    def _make_unique_output_volume_path(self, out_dir: str) -> str:
        # Use a timestamped base name so each run keeps its own output.
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"VolumeOutput_{timestamp}"
        candidate = os.path.join(out_dir, f"{base_name}.mha")

        # If a file exists, append a numeric suffix until we find a free name.
        counter = 1
        while os.path.exists(candidate):
            candidate = os.path.join(out_dir, f"{base_name}_{counter}.mha")
            counter += 1

        # Track the output path so finish handlers know where to load from.
        self._recon_output_path = candidate
        return candidate

    # Summary:
    # - Ensure the reconstruction output log dialog exists and is visible.
    # - Input: `self`.
    # - Returns: None.
    def _open_recon_log_dialog(self) -> None:
        # Create the dialog only once per run so we can update it as output streams in.
        if self._recon_log_dialog is None:
            dialog = QDialog(self)
            dialog.setObjectName("volumeReconLogDialog")
            # UI state change: label the dialog so users know what it shows.
            dialog.setWindowTitle("Volume Reconstruction Output")
            dialog.setModal(False)
            dialog.setWindowModality(Qt.NonModal)
            dialog.resize(600, 150)

            layout = QVBoxLayout(dialog)
            text_edit = QTextEdit(dialog)
            text_edit.setObjectName("volumeReconLogTextEdit")
            # UI state change: keep the log view read-only.
            text_edit.setReadOnly(True)
            text_edit.setLineWrapMode(QTextEdit.NoWrap)
            layout.addWidget(text_edit)
            dialog.setLayout(layout)

            # Signal connection: track when the user closes the dialog.
            dialog.finished.connect(self._on_volumeReconLogDialog_finished)

            self._recon_log_dialog = dialog
            self._recon_log_text = text_edit

        # UI state change: clear old text before showing a new run.
        self._reset_recon_log_state()

        # UI state change: show the dialog non-modally and bring it to the front.
        self._recon_log_dialog.show()
        self._recon_log_dialog.raise_()
        self._recon_log_dialog.activateWindow()

    # Summary:
    # - Close the reconstruction log dialog if it is open.
    # - Input: `self`.
    # - Returns: None.
    def _close_recon_log_dialog(self) -> None:
        # UI state change: close the dialog so it disappears after finishing.
        if self._recon_log_dialog is None:
            return
        # Use done() so the dialog emits finished and we can clean references.
        self._recon_log_dialog.done(0)

    # Summary:
    # - Reset reconstruction log buffers and clear the log text view.
    # - Input: `self`.
    # - Returns: None.
    def _reset_recon_log_state(self) -> None:
        # Clear partial line buffers so line-by-line output starts clean.
        self._recon_stdout_line_buffer = ""
        self._recon_stderr_line_buffer = ""

        # UI state change: clear the visible log when the dialog exists.
        if self._recon_log_text is not None:
            self._recon_log_text.clear()

    # Summary:
    # - Append reconstruction output to the log dialog line-by-line.
    # - Input: `self`, `text` (str), `is_stderr` (bool).
    # - Returns: None.
    def _append_recon_log_output(self, text: str, is_stderr: bool) -> None:
        if not text:
            return

        # Keep partial lines so we only append complete lines to the UI.
        if is_stderr:
            pending = self._recon_stderr_line_buffer
        else:
            pending = self._recon_stdout_line_buffer

        combined = pending + text
        lines = combined.splitlines(keepends=True)
        new_buffer = ""

        # Append only complete lines to keep the log readable line-by-line.
        for line in lines:
            if line.endswith("\n") or line.endswith("\r"):
                clean_line = line.rstrip("\r\n")
                if self._recon_log_text is not None:
                    prefix = "[stderr] " if is_stderr else ""
                    self._recon_log_text.append(f"{prefix}{clean_line}")
            else:
                new_buffer = line

        # Store any trailing partial line for the next read.
        if is_stderr:
            self._recon_stderr_line_buffer = new_buffer
        else:
            self._recon_stdout_line_buffer = new_buffer

        # UI state change: keep the log scrolled to the newest output.
        if self._recon_log_text is not None:
            self._recon_log_text.moveCursor(QTextCursor.End)
            self._trim_recon_log_text()

    # Summary:
    # - Trim the log dialog text to a maximum character count.
    # - Input: `self`.
    # - Returns: None.
    def _trim_recon_log_text(self) -> None:
        # Guard against trimming when the dialog is not visible.
        if self._recon_log_text is None:
            return

        document = self._recon_log_text.document()
        current_count = document.characterCount()
        if current_count <= RECON_LOG_TEXT_LIMIT:
            return

        # Remove the oldest characters so the log does not grow unbounded.
        excess = current_count - RECON_LOG_TEXT_LIMIT
        cursor = QTextCursor(document)
        cursor.movePosition(QTextCursor.Start)
        cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, excess)
        cursor.removeSelectedText()

    # Summary:
    # - Slot function that handles the log dialog being closed.
    # - Input: `self`, `result` (int).
    # - Returns: None.
    def _on_volumeReconLogDialog_finished(self, result: int) -> None:
        # This is a slot function for dialog close events.
        _ = result
        if self._recon_log_dialog is None:
            return

        # Allow reconstruction to continue; just drop UI references.
        self._recon_log_dialog.deleteLater()
        self._recon_log_dialog = None
        self._recon_log_text = None

    # Summary:
    # - Start the external reconstruction process asynchronously.
    # - Input: `self`, `config_path` (str), `seq_path` (str), `out_dir` (str).
    # - Returns: None.
    def _start_reconstruct(self, config_path: str, seq_path: str, out_dir: str) -> None:
        # Prefer an environment override so deployments can set the executable location.
        env_override = os.getenv(VOLUME_RECONSTRUCTOR_EXE_ENV)
        exe_path = env_override.strip() if env_override else DEFAULT_VOLUME_RECONSTRUCTOR_EXE
        exe_path = os.path.abspath(exe_path)

        # Validation: ensure the executable exists before we try to start it.
        if not os.path.isfile(exe_path):
            QMessageBox.warning(
                self,
                "Reconstructor Not Found",
                f"The executable was not found:\n{exe_path}",
            )
            return

        # Build a unique output path for this reconstruction run.
        output_path = self._make_unique_output_volume_path(out_dir)

        # Reset diagnostic buffers so each run has clean logs.
        self._recon_stdout = ""
        self._recon_stderr = ""

        # UI state change: open the live log dialog for this reconstruction run.
        self._open_recon_log_dialog()

        # Create a fresh QProcess so signals are scoped to this run.
        self._recon_process = QProcess(self)
        self._recon_process.setObjectName("volumeReconstructProcess")

        # Signal connection: append stdout to the diagnostics buffer.
        self._recon_process.readyReadStandardOutput.connect(
            self._on_volumeReconstructProcess_stdout_ready
        )
        # Signal connection: append stderr to the diagnostics buffer.
        self._recon_process.readyReadStandardError.connect(
            self._on_volumeReconstructProcess_stderr_ready
        )
        # Signal connection: handle process-level errors (start/crash).
        self._recon_process.errorOccurred.connect(
            self._on_volumeReconstructProcess_error_occurred
        )
        # Signal connection: handle process completion and load the output.
        self._recon_process.finished.connect(
            self._on_volumeReconstructProcess_finished
        )

        # Build the arguments exactly as expected by the CLI tool.
        arguments = [
            f"--config-file={config_path}",
            f"--source-seq-file={seq_path}",
            f"--output-volume-file={output_path}",
            "--image-to-reference-transform=ImageToReference",
            "--disable-compression",
        ]

        # UI state change: lock inputs while the external process runs.
        self._set_recon_ui_running(True)
        # Track the running state so the button can stop the process.
        self._recon_state = "running"

        # Async start: launch the process without blocking the UI thread.
        self._recon_process.start(exe_path, arguments)

    # Summary:
    # - Request the running reconstruction process to stop.
    # - Input: `self`.
    # - Returns: None.
    def _stop_reconstruct(self) -> None:
        # Guard against stop requests when no process exists.
        if self._recon_process is None:
            return
        # Guard against stop requests when the process already exited.
        if self._recon_process.state() == QProcess.NotRunning:
            return
        # Async stop: request termination without blocking the UI.
        self._recon_process.terminate()

    # Summary:
    # - Slot function that starts or stops the reconstruction process.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_volume_reconstruct_clicked(self) -> None:
        # Toggle behavior: stop when running, start when idle.
        if self._recon_state == "running":
            # User intent: stop the external process if it is running.
            self._stop_reconstruct()
            return

        # Validate inputs before starting the external tool.
        inputs = self._validate_recon_inputs()
        if inputs is None:
            return

        config_path, seq_path, out_dir = inputs
        # Start reconstruction asynchronously using the validated inputs.
        self._start_reconstruct(config_path, seq_path, out_dir)

    # Summary:
    # - Slot function that buffers reconstruction stdout output.
    # - Input: `self`.
    # - Returns: None.
    def _on_volumeReconstructProcess_stdout_ready(self) -> None:
        # Guard against signals firing after cleanup.
        if self._recon_process is None:
            return

        # Read any available stdout bytes from the process.
        data = self._recon_process.readAllStandardOutput()
        if not data:
            return

        # Decode and append so we can show diagnostics on failure.
        text = bytes(data).decode(errors="replace")
        self._recon_stdout += text

        # UI update: stream stdout into the live log dialog.
        self._append_recon_log_output(text, is_stderr=False)

        # Keep only the latest chunk to avoid unbounded memory growth.
        if len(self._recon_stdout) > RECON_OUTPUT_BUFFER_LIMIT:
            self._recon_stdout = self._recon_stdout[-RECON_OUTPUT_BUFFER_LIMIT:]

    # Summary:
    # - Slot function that buffers reconstruction stderr output.
    # - Input: `self`.
    # - Returns: None.
    def _on_volumeReconstructProcess_stderr_ready(self) -> None:
        # Guard against signals firing after cleanup.
        if self._recon_process is None:
            return

        # Read any available stderr bytes from the process.
        data = self._recon_process.readAllStandardError()
        if not data:
            return

        # Decode and append so we can show diagnostics on failure.
        text = bytes(data).decode(errors="replace")
        self._recon_stderr += text

        # UI update: stream stderr into the live log dialog.
        self._append_recon_log_output(text, is_stderr=True)

        # Keep only the latest chunk to avoid unbounded memory growth.
        if len(self._recon_stderr) > RECON_OUTPUT_BUFFER_LIMIT:
            self._recon_stderr = self._recon_stderr[-RECON_OUTPUT_BUFFER_LIMIT:]

    # Summary:
    # - Slot function that handles reconstruction process errors.
    # - Input: `self`, `process_error` (QProcess.ProcessError).
    # - Returns: None.
    def _on_volumeReconstructProcess_error_occurred(
        self, process_error: QProcess.ProcessError
    ) -> None:
        # Normalize state first so the UI is re-enabled even on early failures.
        self._recon_state = "idle"
        self._set_recon_ui_running(False)
        # UI state change: close the live log dialog when the process errors out.
        self._close_recon_log_dialog()

        # Collect diagnostics to help the user debug failed starts.
        exe_path = ""
        error_text = "Unknown error"
        if self._recon_process is not None:
            exe_path = self._recon_process.program()
            error_text = self._recon_process.errorString()

        stderr_tail = self._recon_stderr[-RECON_OUTPUT_BUFFER_LIMIT:]
        if not stderr_tail:
            stderr_tail = "(no stderr captured)"

        # UI state change: show a warning with the process error details.
        QMessageBox.warning(
            self,
            "Reconstruction Error",
            "\n".join(
                [
                    f"Executable: {exe_path}",
                    f"Error: {process_error} ({error_text})",
                    "Stderr (tail):",
                    stderr_tail,
                ]
            ),
        )

        # Cleanup: release the process object after handling the error.
        self._cleanup_recon_process()

    # Summary:
    # - Slot function that handles reconstruction completion.
    # - Input: `self`, `exit_code` (int), `exit_status` (QProcess.ExitStatus).
    # - Returns: None.
    def _on_volumeReconstructProcess_finished(
        self, exit_code: int, exit_status: QProcess.ExitStatus
    ) -> None:
        # Guard against double handling after an error cleanup.
        if self._recon_process is None:
            return

        # Restore UI state regardless of success or failure.
        self._recon_state = "idle"
        self._set_recon_ui_running(False)
        # UI state change: close the live log dialog now that the process ended.
        self._close_recon_log_dialog()

        # Validate normal exit before attempting to load outputs.
        if exit_status != QProcess.NormalExit or exit_code != 0:
            stderr_tail = self._recon_stderr[-RECON_OUTPUT_BUFFER_LIMIT:]
            if not stderr_tail:
                stderr_tail = "(no stderr captured)"
            QMessageBox.warning(
                self,
                "Reconstruction Failed",
                "\n".join(
                    [
                        f"Exit status: {exit_status}",
                        f"Exit code: {exit_code}",
                        "Stderr (tail):",
                        stderr_tail,
                    ]
                ),
            )
            self._cleanup_recon_process()
            return

        output_path = self._recon_output_path
        if not output_path or not os.path.isfile(output_path):
            QMessageBox.warning(
                self,
                "Output Missing",
                f"The output volume file was not found:\n{output_path}",
            )
            self._cleanup_recon_process()
            return

        # UI state change: show the new output path in the volume line edit.
        self.ui.lineEdit_volume_volfile.setText(output_path)
        # Load and display the output volume using the existing pipeline.
        self._load_and_display_volume(output_path)

        # Cleanup: release the process object once we have handled the output.
        self._cleanup_recon_process()

    # Summary:
    # - Cleanup the reconstruction QProcess to avoid stale signal emissions.
    # - Input: `self`.
    # - Returns: None.
    def _cleanup_recon_process(self) -> None:
        # Guard against cleanup when the process is already cleared.
        if self._recon_process is None:
            return
        # Allow Qt to delete the process safely on the event loop.
        self._recon_process.deleteLater()
        self._recon_process = None


# Summary:
# - Application entry point that shows the VolumeWidget window.
# - Input: None (reads command-line args via sys.argv).
# - Returns: None.
def main() -> None:
    # Import sys here to keep module imports focused for the widget.
    import sys

    # Create and run the Qt application.
    app = QApplication(sys.argv)
    window = VolumeWidget()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
