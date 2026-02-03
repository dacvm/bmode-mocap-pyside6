"""PySide6 widget that wires up the Volume UI file selection controls."""

# Standard library helpers for path normalization.
import os
from typing import Optional

# Qt widgets for the main window and file dialogs.
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

# Generated UI class from Qt Designer (do not edit the _ui.py file).
from volume_ui import Ui_Form as Ui_Volume

from helpers.mha_reader import MhaReader
from helpers.mha_volume import MhaVolume

DEFAULT_CONFIG_FILE = (
    "configs/PlusDeviceSet_fCal_Epiphan_NDIPolaris_RadboudUMC_20241219_150400.xml"
)
DEFAULT_SEQUENCE_FILE = "seqs/SequenceRecording_2024-12-20_14-47-41.mha"
DEFAULT_OUTPUT_DIR = "seqs/"
DEFAULT_VOLUME_FILE = "seqs/VolumeOutput_2024-12-20_14-48-05.mha"


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

        # Keep a reader instance so we can reuse it for multiple loads.
        self._mha_reader = MhaReader()
        # Track the last loaded volume so other actions can reuse it later.
        self._mha_volume: Optional[MhaVolume] = None

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

        # Load the .mha file into a MhaVolume container (no reconstruction yet).
        try:
            self._mha_volume = self._mha_reader.read(volfile_path, use_memmap=True)
        except ValueError as exc:
            # UI state change: surface parsing errors without crashing the UI.
            QMessageBox.warning(self, "Volume Load Failed", str(exc))
            return


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
