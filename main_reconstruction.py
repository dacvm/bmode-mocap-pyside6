"""Main window wrapper that embeds B-mode, Mocap, and Volume widgets."""

# Standard library helpers for filesystem normalization and app exit codes.
from datetime import datetime
import os
import sys
from typing import Optional

# Qt core helpers for queued signal connections and typed slots.
from PySide6.QtCore import Qt, Slot
# Qt widgets for main window, dialogs, and message boxes.
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox

# Coupler that matches image and rigid body packets by timestamp.
from helpers.coupled_stream_controller import CoupledStreamController
# Generated UI class from Qt Designer (do not edit the generated file).
from app_reconstruction_ui import Ui_MainWindow
# Sequence writer that exports coupled packets into Plus-style .mha files.
from mha_writer import MhaWriter
# Custom widgets that will be embedded into the main window layouts.
from main_bmode import BModeWidget
from main_mocap import MocapWidget
from main_volume import VolumeWidget

COUPLED_MAXDIFF_IMAGEPOSE_MS = 120
COUPLED_MOCAP_BUFFER_MAXLEN = 500


# Summary:
# - Main window wrapper that hosts the reconstruction UI and embeds custom widgets.
# - What it does: Builds the generated UI, inserts B-mode/Mocap/Volume widgets, and
#   implements coupled-record directory interactions.
# - Input: `self` (new window instance).
# - Returns: A fully initialized QMainWindow subclass instance.
class MainWindow(QMainWindow):
    # Summary:
    # - Initialize the main window and wire up the UI behavior.
    # - What it does: Creates the generated UI, embeds custom widgets, adjusts layout
    #   spacing, and connects button signals to slot handlers.
    # - Input: `self` (the new window instance).
    # - Returns: None.
    def __init__(self) -> None:
        super().__init__()

        # Build the generated UI so all Qt Designer widgets are available.
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Make embedded layouts tight so custom widgets fill their group boxes.
        self.ui.verticalLayout_bmode.setContentsMargins(0, 0, 0, 0)
        self.ui.verticalLayout_bmode.setSpacing(4)
        self.ui.verticalLayout_mocap.setContentsMargins(0, 0, 0, 0)
        self.ui.verticalLayout_mocap.setSpacing(4)
        self.ui.verticalLayout_volume.setContentsMargins(0, 0, 0, 0)
        self.ui.verticalLayout_volume.setSpacing(4)

        # Create custom widgets and keep references to avoid garbage collection.
        self._bmode_widget = BModeWidget()
        self._mocap_widget = MocapWidget()
        self._volume_widget = VolumeWidget()

        # Add custom widgets into their assigned layouts.
        self.ui.verticalLayout_bmode.addWidget(self._bmode_widget)
        self.ui.verticalLayout_mocap.addWidget(self._mocap_widget)
        self.ui.verticalLayout_volume.addWidget(self._volume_widget)

        # UI state change: keep status labels in a known initial state on startup.
        self.ui.label_status_bmode.setText("Disconnected")
        self.ui.label_status_mocap.setText("Disconnected")

        # Track the latest debug values so one status line can show coupling health.
        self._latest_coupled_diff_ms: Optional[int] = None
        self._latest_coupler_stats: dict = {}
        # Track active record mode and writer so all stop paths can finalize safely.
        self._mha_writer: Optional[MhaWriter] = None
        self._active_record_mode: Optional[str] = None

        # Create the software coupler for image+pose sample-and-hold matching.
        self._coupler = CoupledStreamController(
            maxdiff_imagepose_ms=COUPLED_MAXDIFF_IMAGEPOSE_MS,
            mocap_buffer_maxlen=COUPLED_MOCAP_BUFFER_MAXLEN,
        )

        # Signal connection: mirror B-mode stream state into the reconstruction status label.
        self._bmode_widget._proxy.state_changed.connect(
            self._on_bmodeStreamProxy_state_changed
        )
        # Signal connection: mirror Mocap stream state into the reconstruction status label.
        self._mocap_widget._stream_proxy.state_changed.connect(
            self._on_mocapStreamProxy_state_changed
        )
        # Signal connection: forward image packets into the coupler on a queued connection.
        self._bmode_widget.sig_image_packet.connect(
            self._coupler.on_image_packet,
            Qt.QueuedConnection,
        )
        # Signal connection: forward rigid body packets into the coupler on a queued connection.
        self._mocap_widget.sig_rigidbody_packet.connect(
            self._coupler.on_rigidbody_packet,
            Qt.QueuedConnection,
        )
        # Signal connection: show coupled packet timing for debug visibility.
        self._coupler.sig_coupled_packet.connect(
            self._on_coupledStreamController_coupled_packet,
            Qt.QueuedConnection,
        )
        # Signal connection: show coupling drop counters for debug visibility.
        self._coupler.sig_stats.connect(
            self._on_coupledStreamController_stats_updated,
            Qt.QueuedConnection,
        )

        # Signal connection: browse for an output directory.
        self.ui.pushButton_coupledrecord_recorddirBrowse.clicked.connect(
            self._on_pushButton_coupledrecord_recorddirBrowse_clicked
        )
        # Signal connection: clear the output directory.
        self.ui.pushButton_coupledrecord_recorddirClear.clicked.connect(
            self._on_pushButton_coupledrecord_recorddirClear_clicked
        )
        # Signal connection: run the record stream
        self.ui.pushButton_coupledrecord_recordStream.clicked.connect(
            self._on_pushButton_coupledrecord_recordStream_clicked
        )
        # UI state change: initialize coupled debug line in the status bar.
        self._update_coupled_debug_status()

    # Summary:
    # - Slot function for B-mode stream-state updates from the embedded B-mode widget.
    # - What it does: Keeps the reconstruction status label synchronized with stream start/stop.
    # - Input: `self`, `is_running` (bool), `message` (str).
    # - Returns: None.
    def _on_bmodeStreamProxy_state_changed(self, is_running: bool, message: str) -> None:
        # Slot function: react to stream state changes emitted by the B-mode widget proxy.
        _ = message

        # UI state change: show the active stream state in the B-mode status label.
        if is_running:
            self.ui.label_status_bmode.setText("Streaming")
            return

        # UI state change: restore idle state whenever B-mode streaming stops or fails.
        self.ui.label_status_bmode.setText("Disconnected")
        # Stop coupled recording when one source drops so sessions stay valid.
        if self._coupler.is_recording():
            had_active_mha_writer = (
                self._active_record_mode == ".mha" and self._mha_writer is not None
            )
            self._coupler.stop_recording()
            # Finalize any active .mha recording so payload and header stay consistent.
            final_path = self._finalize_active_mha_recording(
                show_success_message=False,
                show_error_dialog=True,
            )
            if final_path:
                self.statusBar().showMessage(
                    f"B-mode stream stopped; partial .mha saved: {final_path}",
                    9000,
                )
            elif not had_active_mha_writer:
                self.statusBar().showMessage(
                    "B-mode stream stopped; coupled recording was ended.",
                    9000,
                )
            # UI state change: restore record button text after forced stop.
            self.ui.pushButton_coupledrecord_recordStream.setText("Record")

    # Summary:
    # - Slot function for mocap stream-state updates from the embedded mocap widget.
    # - What it does: Keeps the reconstruction status label synchronized with QTM stream start/stop.
    # - Input: `self`, `is_running` (bool), `message` (str).
    # - Returns: None.
    def _on_mocapStreamProxy_state_changed(self, is_running: bool, message: str) -> None:
        # Slot function: react to stream state changes emitted by the mocap widget proxy.
        _ = message

        # UI state change: show the active stream state in the mocap status label.
        if is_running:
            self.ui.label_status_mocap.setText("Streaming")
            return

        # UI state change: restore idle state whenever mocap streaming stops or fails.
        self.ui.label_status_mocap.setText("Disconnected")
        # Stop coupled recording when one source drops so sessions stay valid.
        if self._coupler.is_recording():
            had_active_mha_writer = (
                self._active_record_mode == ".mha" and self._mha_writer is not None
            )
            self._coupler.stop_recording()
            # Finalize any active .mha recording so payload and header stay consistent.
            final_path = self._finalize_active_mha_recording(
                show_success_message=False,
                show_error_dialog=True,
            )
            if final_path:
                self.statusBar().showMessage(
                    f"Mocap stream stopped; partial .mha saved: {final_path}",
                    9000,
                )
            elif not had_active_mha_writer:
                self.statusBar().showMessage(
                    "Mocap stream stopped; coupled recording was ended.",
                    9000,
                )
            # UI state change: restore record button text after forced stop.
            self.ui.pushButton_coupledrecord_recordStream.setText("Record")

    # Summary:
    # - Slot function for coupled packet events from the CoupledStreamController.
    # - What it does: caches the latest diff and refreshes debug status text.
    # - Input: `self`, `image_ts_ms` (int), `image_data` (object), `rigidbody_ts_ms` (int),
    #   `rigidbody_data` (object), `diff_ms` (int).
    # - Returns: None.
    @Slot(int, object, int, object, int)
    def _on_coupledStreamController_coupled_packet(
        self,
        image_ts_ms: int,
        image_data: object,
        rigidbody_ts_ms: int,
        rigidbody_data: object,
        diff_ms: int,
    ) -> None:
        # Cache the latest diff so status/debug text can show coupling quality.
        self._latest_coupled_diff_ms = int(diff_ms)
        self._update_coupled_debug_status()
        # During active .mha recording, append this accepted coupled packet to disk.
        if (
            self._coupler.is_recording()
            and self._active_record_mode == ".mha"
            and self._mha_writer is not None
        ):
            try:
                self._mha_writer.append_coupled_packet(
                    image_ts_ms=image_ts_ms,
                    image_data=image_data,
                    rigidbody_ts_ms=rigidbody_ts_ms,
                    rigidbody_data=rigidbody_data,
                )
            except Exception as exc:
                # Stop and finalize on write failure to avoid leaving temp payload files behind.
                if self._coupler.is_recording():
                    self._coupler.stop_recording()
                self._finalize_active_mha_recording(
                    show_success_message=False,
                    show_error_dialog=False,
                )
                # UI state change: reflect that recording is no longer active.
                self.ui.pushButton_coupledrecord_recordStream.setText("Record")
                QMessageBox.warning(
                    self,
                    "MHA Write Error",
                    f"Coupled recording stopped because writing failed:\n{exc}",
                )
                self.statusBar().showMessage(
                    f"Coupled recording stopped due to .mha write error: {exc}",
                    9000,
                )

    # Summary:
    # - Slot function for coupling stats updates from the CoupledStreamController.
    # - What it does: caches the latest counters and refreshes debug status text.
    # - Input: `self`, `stats` (dict).
    # - Returns: None.
    @Slot(dict)
    def _on_coupledStreamController_stats_updated(self, stats: dict) -> None:
        # Validation/transform: guard against unexpected payloads before caching.
        if not isinstance(stats, dict):
            return
        self._latest_coupler_stats = dict(stats)
        self._update_coupled_debug_status()

    # Summary:
    # - Rebuild the coupled-stream debug text in the status bar.
    # - What it does: combines latest diff and counters so users can quickly verify coupling health.
    # - Input: `self`.
    # - Returns: None.
    def _update_coupled_debug_status(self) -> None:
        # Read stats with safe defaults so the message still renders before first packet.
        count_coupled = int(self._latest_coupler_stats.get("count_coupled", 0))
        count_dropped_no_pose = int(
            self._latest_coupler_stats.get("count_dropped_no_pose", 0)
        )
        count_dropped_stale = int(
            self._latest_coupler_stats.get("count_dropped_stale", 0)
        )
        is_recording = bool(self._latest_coupler_stats.get("is_recording", False))

        # Render diff as "N/A" until the first coupled packet arrives.
        if self._latest_coupled_diff_ms is None:
            diff_text = "N/A"
        else:
            diff_text = f"{self._latest_coupled_diff_ms} ms"

        # UI state change: show one compact debug line for coupling success and drop reasons.
        state_text = "REC" if is_recording else "IDLE"
        self.statusBar().showMessage(
            "Coupled streaming "
            f"[ {state_text} ] Diff = {diff_text} | "
            f"Data coupled = {count_coupled} | "
            f"Data dropped (no match) = {count_dropped_no_pose} | "
            f"Data dropped (too old) = {count_dropped_stale}"
        )


    # Summary:
    # - Slot function for the record directory clear button.
    # - What it does: Clears the record directory line edit so the user can start over.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_coupledrecord_recorddirClear_clicked(self) -> None:
        # Slot function: react to the clear button click.
        # UI state change: clear the text to show the directory is unset.
        self.ui.lineEdit_coupledrecord_recorddir.clear()

    # Summary:
    # - Slot function for the record directory browse button.
    # - What it does: Opens a folder picker and normalizes the chosen directory path.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_coupledrecord_recorddirBrowse_clicked(self) -> None:
        # Slot function: react to the browse button click.
        # Ask the user for a directory so recordings go to a known location.
        chosen_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Record Directory",
        )

        # Only update the UI when a folder is actually chosen.
        if chosen_dir:
            # Normalize the path so downstream code gets a clean absolute location.
            path = os.path.abspath(os.path.expanduser(chosen_dir))
            # UI state change: show the normalized path to the user.
            self.ui.lineEdit_coupledrecord_recorddir.setText(path)

    # Summary:
    # - Slot function for the Record button.
    # - What it does: Toggles coupled recording, validates prerequisites on start,
    #   and controls the CoupledStreamController recording gate.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_coupledrecord_recordStream_clicked(self) -> None:
        # Slot function: react to the Record button click.
        # Toggle behavior: stop immediately if coupled recording is currently active.
        if self._coupler.is_recording():
            had_active_mha_writer = (
                self._active_record_mode == ".mha" and self._mha_writer is not None
            )
            self._coupler.stop_recording()
            # Finalize active .mha output now that recording has ended.
            final_path = self._finalize_active_mha_recording(
                show_success_message=False,
                show_error_dialog=True,
            )
            # UI state change: restore button text when recording stops.
            self.ui.pushButton_coupledrecord_recordStream.setText("Record")
            if final_path:
                self.statusBar().showMessage(
                    f"Coupled recording saved to: {final_path}",
                    9000,
                )
            elif not had_active_mha_writer:
                self.statusBar().showMessage(
                    "Coupled recording stopped.",
                    5000,
                )
            return

        # Validation: ensure a record directory is set before proceeding.
        record_dir = self.ui.lineEdit_coupledrecord_recorddir.text().strip()
        if not record_dir:
            QMessageBox.warning(
                self,
                "Record Directory Missing",
                "Please choose a record directory before recording.",
            )
            return

        # Validation: require both sources to be streaming before recording.
        bmode_state = self.ui.label_status_bmode.text().strip().lower()
        mocap_state = self.ui.label_status_mocap.text().strip().lower()
        if "streaming" not in bmode_state or "streaming" not in mocap_state:
            QMessageBox.warning(
                self,
                "Streaming Required",
                "Start both B-mode and Mocap streaming before recording.",
            )
            return

        # Defensive check: radio buttons might not exist yet in the UI.
        if not hasattr(self.ui, "radioButton_imagecsv") or not hasattr(
            self.ui, "radioButton_mha"
        ):
            QMessageBox.warning(
                self,
                "Record Mode Missing",
                "Record mode controls are not available in the current UI.",
            )
            return

        # Normalize/create output directory early so writer startup errors are explicit.
        normalized_record_dir = os.path.abspath(os.path.expanduser(record_dir))
        try:
            os.makedirs(normalized_record_dir, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Record Directory Error",
                f"Could not prepare record directory:\n{exc}",
            )
            return

        # Derive selected mode and initialize writer only for .mha mode.
        if self.ui.radioButton_imagecsv.isChecked():
            selected_mode = "image + .csv"
            self._mha_writer = None
            self._active_record_mode = selected_mode
        elif self.ui.radioButton_mha.isChecked():
            selected_mode = ".mha"
            output_mha_path = self._build_mha_output_path(normalized_record_dir)
            writer = MhaWriter(invert_transforms=False)
            try:
                writer.start(output_mha_path)
            except Exception as exc:
                # Cleanup state when startup fails before recording can start.
                self._mha_writer = None
                self._active_record_mode = None
                QMessageBox.warning(
                    self,
                    "MHA Start Failed",
                    f"Could not start .mha recording:\n{exc}",
                )
                return
            self._mha_writer = writer
            self._active_record_mode = ".mha"
        else:
            selected_mode = "unknown"
            self._mha_writer = None
            self._active_record_mode = None

        # Start coupling session so incoming image packets produce coupled packets.
        self._coupler.start_recording()
        self._latest_coupled_diff_ms = None
        # UI state change: show stop intent while coupled recording is active.
        self.ui.pushButton_coupledrecord_recordStream.setText("Stop Recording")
        # UI state change: provide immediate debug text for active mode/session.
        self.statusBar().showMessage(
            f"Coupled recording started ({selected_mode}). Waiting for coupled packets..."
        )

    # Summary:
    # - Build a timestamped output path for one coupled sequence `.mha` file.
    # - What it does: Normalizes the record directory and creates a deterministic Plus-like filename.
    # - Input: `self`, `record_dir` (str).
    # - Returns: Full `.mha` output path (str).
    def _build_mha_output_path(self, record_dir: str) -> str:
        # Normalize path so all writer calls receive a stable absolute directory.
        normalized_dir = os.path.abspath(os.path.expanduser(record_dir))
        timestamp_text = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"SequenceRecording_{timestamp_text}.mha"
        return os.path.join(normalized_dir, filename)

    # Summary:
    # - Finalize and clear the active `.mha` writer session if one is running.
    # - What it does: Calls `MhaWriter.finalize()` with error handling, updates status/UI feedback,
    #   and always clears writer state so future recordings start cleanly.
    # - Input: `self`, `show_success_message` (bool), `show_error_dialog` (bool).
    # - Returns: Final path on success, otherwise None (Optional[str]).
    def _finalize_active_mha_recording(
        self,
        show_success_message: bool,
        show_error_dialog: bool,
    ) -> Optional[str]:
        final_path = None

        # No-op when current record mode is not .mha or writer was never created.
        if self._active_record_mode != ".mha" or self._mha_writer is None:
            self._mha_writer = None
            self._active_record_mode = None
            return None

        try:
            final_path = self._mha_writer.finalize()
            if show_success_message:
                # UI state change: confirm final file path for quick operator verification.
                self.statusBar().showMessage(
                    f"Coupled .mha saved: {final_path}",
                    9000,
                )
        except Exception as exc:
            # Surface finalize failures because they can indicate incomplete recording output.
            self.statusBar().showMessage(
                f"Failed to finalize coupled .mha recording: {exc}",
                9000,
            )
            if show_error_dialog:
                QMessageBox.warning(
                    self,
                    "MHA Finalize Error",
                    f"Could not finalize .mha recording:\n{exc}",
                )
        finally:
            # Always clear mode/writer references to avoid stale state on the next session.
            self._mha_writer = None
            self._active_record_mode = None

        return final_path

    # Summary:
    # - Handle the main-window close event.
    # - What it does: Forces both status labels back to disconnected so UI state is consistent on shutdown.
    # - Input: `self`, `event` (Qt close event object).
    # - Returns: None.
    def closeEvent(self, event) -> None:
        # UI state change: reset both labels during shutdown for deterministic final state.
        self.ui.label_status_bmode.setText("Disconnected")
        self.ui.label_status_mocap.setText("Disconnected")
        # Stop coupling so queued packets are ignored during window teardown.
        if self._coupler.is_recording():
            self._coupler.stop_recording()
            # Finalize best-effort on close to keep the current recording recoverable.
            self._finalize_active_mha_recording(
                show_success_message=False,
                show_error_dialog=False,
            )
        # UI state change: restore record button text on shutdown.
        self.ui.pushButton_coupledrecord_recordStream.setText("Record")

        # Delegate to Qt's default close handling after local cleanup.
        super().closeEvent(event)
        event.accept()


# Summary:
# - Entry point for launching the reconstruction main window.
# - What it does: Creates the QApplication, shows the MainWindow, and starts the Qt loop.
# - Input: None.
# - Returns: Process exit code (int).
def main() -> int:
    # Create the Qt application instance so widgets can be shown.
    app = QApplication(sys.argv)
    # Create and show the main window wrapper.
    window = MainWindow()
    window.show()    
    # Run the Qt event loop and return its exit code.
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
