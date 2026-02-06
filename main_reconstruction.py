"""Main window wrapper that embeds B-mode, Mocap, and Volume widgets."""

# Standard library helpers for filesystem normalization and app exit codes.
import os
import sys

# Qt widgets for main window, dialogs, and message boxes.
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox

# Generated UI class from Qt Designer (do not edit the generated file).
from app_reconstruction_ui import Ui_MainWindow
# Custom widgets that will be embedded into the main window layouts.
from main_bmode import BModeWidget
from main_mocap import MocapWidget
from main_volume import VolumeWidget


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

        # Signal connection: mirror B-mode stream state into the reconstruction status label.
        self._bmode_widget._proxy.state_changed.connect(
            self._on_bmodeStreamProxy_state_changed
        )
        # Signal connection: mirror Mocap stream state into the reconstruction status label.
        self._mocap_widget._stream_proxy.state_changed.connect(
            self._on_mocapStreamProxy_state_changed
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
    # - What it does: Validates record directory and streaming status, then prints
    #   the selected recording mode.
    # - Input: `self`.
    # - Returns: None.
    def _on_pushButton_coupledrecord_recordStream_clicked(self) -> None:
        # Slot function: react to the Record button click.
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
            print("radio buttons not found in Ui_MainWindow")
            return

        # Emit the selected mode so downstream logic can be wired later.
        if self.ui.radioButton_imagecsv.isChecked():
            print("image+csv is selected")
        elif self.ui.radioButton_mha.isChecked():
            print("mha is selected")

    # Summary:
    # - Handle the main-window close event.
    # - What it does: Forces both status labels back to disconnected so UI state is consistent on shutdown.
    # - Input: `self`, `event` (Qt close event object).
    # - Returns: None.
    def closeEvent(self, event) -> None:
        # UI state change: reset both labels during shutdown for deterministic final state.
        self.ui.label_status_bmode.setText("Disconnected")
        self.ui.label_status_mocap.setText("Disconnected")

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
