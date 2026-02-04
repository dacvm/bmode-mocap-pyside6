"""PySide6 widget that wires up the Volume UI file selection controls."""

# Standard library helpers for path normalization and threading.
import os
import threading
from typing import Optional

# Third-party imports for plotting and numeric helpers.
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Signal, QTimer

# Qt widgets for the main window and file dialogs.
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

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
    # - Slot function that loads the selected volume MHA file.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_volume_volload_clicked(self) -> None:
        # Read the volume file path from the line edit and validate it first.
        volfile_text = self.ui.lineEdit_volume_volfile.text().strip()
        if not volfile_text:
            # UI state change: notify the user the path is required.
            QMessageBox.warning(self, "Missing Volume File", "Select a volume .mha file first.")
            return

        volfile_path = os.path.abspath(volfile_text)
        if not os.path.isfile(volfile_path):
            # UI state change: notify the user when the file is not found.
            QMessageBox.warning(
                self,
                "Volume File Not Found",
                f"The file does not exist:\n{volfile_path}",
            )
            return

        # Load the .mha file into a MhaVolume container (no geometry build yet).
        try:
            self._mha_volume = self._mha_reader.read(volfile_path, use_memmap=True)
        except ValueError as exc:
            # UI state change: surface parsing errors without crashing the UI.
            QMessageBox.warning(self, "Volume Load Failed", str(exc))
            return

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
