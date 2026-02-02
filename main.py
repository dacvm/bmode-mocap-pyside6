"""Live-preview widget that streams from the selected USB camera into the Qt UI."""

# Standard library helpers for launching a Qt application and accepting CLI args.
import sys
from pathlib import Path
# Optional typing helper preserved for future extensions or annotations.
from typing import Optional

# Third-party libs for video capture (OpenCV) and Qt widgets/layouts.
import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPixmap, QScreen
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox, QWidget

# Generated UI classes that wrap the Qt Designer forms into Python-friendly fields.
from bmode_ui_v2 import Ui_Form as Ui_BModeV2
# Class to parse the Calibration XML.
from helpers.xml_stream_parser import (
    CoordinateDefinitionsExtractor,
    VideoDeviceExtractor,
    XmlStreamParser,
)


# Summary: Main Qt window for the B-mode streaming UI.
# What it does: Builds the widgets from the Designer `.ui`, wires signals, and manages stream state (camera/screen).
# Input: Created with no args; uses `self.ui` widgets and internal state to react to user actions.
# Returns: A QWidget subclass instance that can be shown in a QApplication.
class BModeWidget(QWidget):
    """Lightweight window that simply renders the V2 form."""

    # Summary: Initialize the window and connect UI signals.
    # What it does: Sets up the generated UI, prepares timer/camera state, and hooks buttons/combos to handlers.
    # Input: `self` (the new window instance).
    # Returns: `None` (constructor).
    def __init__(self) -> None:
        super().__init__()  # Call the QWidget constructor to initialize Qt internals.
        self.ui = Ui_BModeV2()  # Instantiate the generated UI helper for the V2 layout.
        self.ui.setupUi(self)  # Wire the widgets from the .ui file to this QWidget instance.

        # Runtime state for streaming so we can start/stop cleanly.
        self._camera: Optional[cv2.VideoCapture] = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._grab_frame)
        self._stream_source: Optional[str] = None
        self._screen_rect: Optional[tuple[int, int, int, int]] = None
        self._screen: Optional[QScreen] = None

        # Fill the stream port dropdown with any live camera indices we can open.
        self._populate_cameras()

        # When the stream option changes, we might need to disable the port selector.
        self.ui.comboBox_bmode_streamOption.currentTextChanged.connect(
            self._on_comboBox_bmode_streamOption_changed
        )
        # Apply the current selection immediately so the GUI is consistent on startup.
        self._on_comboBox_bmode_streamOption_changed(self.ui.comboBox_bmode_streamOption.currentText())

        # Trigger the stream-opening flow when the user clicks the connect button.
        self.ui.pushButton_bmode_openStream.clicked.connect(self._on_pushButton_bmode_openStream_clicked)
        # Let the user browse for a calibration XML when they click the browse button.
        self.ui.pushButton_bmode_calibBrowse.clicked.connect(
            self._on_pushButton_bmode_calibBrowse_clicked
        )
        # Reset the calibration path and any related state when the clear button is clicked.
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

    # Summary: Update UI when the stream option changes.
    # What it does: Enables/disables the camera port selector depending on whether we stream a camera or the screen.
    # Input: `self`, `text` (the current stream option label from the combo box).
    # Returns: `None`.
    def _on_comboBox_bmode_streamOption_changed(self, text: str) -> None:
        """Disable the port selector when the user picks the Other screen option."""
        # When streaming to the PC screen, we do not need a camera index, so gray out the selector.
        self.ui.comboBox_bmode_streamPort.setEnabled(text != "Stream Screen (This PC)")

    # Summary: Open/close streaming when the user clicks the button.
    # What it does: Validates the current UI selections, then starts streaming or stops it if already running.
    # Input: `self`.
    # Returns: `None`.
    def _on_pushButton_bmode_openStream_clicked(self) -> None:

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

        # Start (or stop) the stream after the warning so the flow continues.
        if self._timer.isActive():
            self._stop_stream()
        else:
            self._start_stream()

    # Summary: Let the user choose a calibration XML file.
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

        # Parse the XML file. On this moment we only care about the
        parser = XmlStreamParser(str(file_path))
        parser.register(VideoDeviceExtractor())
        data = parser.parse()

        # Get the ClipRectangleOrigin and ClipRectangleSize
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
        origin_x, origin_y = map(int, clip_rectangle_origin.split())
        size_width, size_height = map(int, clip_rectangle_size.split())
        self._screen_rect = (origin_x, origin_y, size_width, size_height)

    # Summary: Clear the calibration path and related cached screen settings.
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
        # Clear the selected screen as it may have been tied to the old calibration.
        self._screen = None

    # Summary: Timer callback that grabs the next frame.
    # What it does: Reads from the active source (screen or camera) and refreshes the UI image.
    # Input: `self`.
    # Returns: `None`.
    def _grab_frame(self) -> None:
        """Read a frame from the active source and display it in the label."""
        if self._stream_source == "screen":
            self._grab_screen_frame()
        else:
            self._grab_camera_frame()

    # Summary: Start streaming based on the selected stream option.
    # What it does: Chooses camera streaming or screen streaming, then starts the periodic timer updates.
    # Input: `self`.
    # Returns: `None`.
    def _start_stream(self) -> None:
        """Start the selected stream source and begin pulling frames on a timer."""
        stream_option = self.ui.comboBox_bmode_streamOption.currentText()
        if stream_option == "Stream Image":
            self._start_camera_stream()
        else:
            self._start_screen_stream()

    # Summary: Start streaming frames from the selected camera.
    # What it does: Opens the chosen camera index, updates button text/state, and starts the frame timer.
    # Input: `self`.
    # Returns: `None`.
    def _start_camera_stream(self) -> None:
        """Open the selected camera and begin pulling frames on a timer."""
        # Read the selected camera index (stored as item data).
        selected_index = self.ui.comboBox_bmode_streamPort.currentData()
        if selected_index is None:
            selected_index = self.ui.comboBox_bmode_streamPort.currentIndex()

        # Open the camera using DirectShow for Windows compatibility.
        self._camera = cv2.VideoCapture(int(selected_index), cv2.CAP_DSHOW)
        if not self._camera.isOpened():
            self._camera.release()
            self._camera = None
            QMessageBox.warning(
                self,
                "Camera Error",
                "Failed to open the selected camera. Please try another port.",
            )
            return

        self._stream_source = "camera"
        # Start a timer to grab frames periodically without blocking the UI thread.
        self._timer.start(30)  # ~33 FPS
        self.ui.pushButton_bmode_openStream.setText("Stop Stream")

    # Summary: Grab one frame from the camera and show it.
    # What it does: Reads a frame via OpenCV, converts it to a Qt image, and renders it in the label.
    # Input: `self`.
    # Returns: `None`.
    def _grab_camera_frame(self) -> None:

        if self._camera is None:
            return

        ret, frame = self._camera.read()
        if not ret:
            # If frame read fails, stop to avoid spamming errors.
            self._stop_stream()
            QMessageBox.warning(
                self,
                "Stream Error",
                "The camera stopped sending frames. The stream has been closed.",
            )
            return

        # Convert the OpenCV BGR image into RGB for Qt.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        qimage = QImage(
            rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qimage)
        self._display_pixmap(pixmap)

    # Summary: Start streaming a calibrated region from the local screen.
    # What it does: Requires calibration rectangle data, asks the user which screen to use, then starts the timer.
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

        self._screen = screen
        self._stream_source = "screen"
        self._timer.start(30)  # ~33 FPS
        self.ui.pushButton_bmode_openStream.setText("Stop Stream")

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

    # Summary: Grab one frame from the selected screen region.
    # What it does: Captures the calibrated rectangle from the chosen screen and displays it in the label.
    # Input: `self`.
    # Returns: `None`.
    def _grab_screen_frame(self) -> None:

        if not self._screen_rect:
            return

        screen = self._screen or QGuiApplication.primaryScreen()
        if screen is None or screen not in QGuiApplication.screens():
            self._stop_stream()
            QMessageBox.warning(
                self,
                "Screen Error",
                "The screen is no longer available. The stream has been closed.",
            )
            return

        origin_x, origin_y, size_width, size_height = self._screen_rect
        if size_width <= 0 or size_height <= 0:
            self._stop_stream()
            QMessageBox.warning(
                self,
                "Calibration Error",
                "Invalid clip rectangle size detected. The stream has been closed.",
            )
            return

        pixmap = screen.grabWindow(0, origin_x, origin_y, size_width, size_height)
        if pixmap.isNull():
            self._stop_stream()
            QMessageBox.warning(
                self,
                "Screen Error",
                "Failed to capture the screen region. The stream has been closed.",
            )
            return

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
    # What it does: Stops the timer, releases the camera, clears stored stream state, and resets the button/label.
    # Input: `self`.
    # Returns: `None`.
    def _stop_stream(self) -> None:
        #Stop the timer and release the camera resource.
        if self._timer.isActive():
            self._timer.stop()
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        self._stream_source = None
        self._screen = None
        self.ui.pushButton_bmode_openStream.setText("Open Stream")
        self.ui.label_bmode_image.clear()

    # Summary: Handle the window close event.
    # What it does: Stops streaming so the camera/timer are cleaned up before the app closes.
    # Input: `self`, `event` (the Qt close event object).
    # Returns: `None`.
    def closeEvent(self, event) -> None:
        # Ensure camera/timer resources are released when the window closes.
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
