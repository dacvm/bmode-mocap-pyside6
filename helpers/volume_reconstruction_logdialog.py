"""Dialog that displays live output from volume reconstruction."""

# Standard library helpers for optional typing.
from typing import Optional

# Qt helpers for window modality and text cursor control.
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QDialog, QTextEdit, QVBoxLayout, QWidget

RECON_LOG_TEXT_LIMIT = 25000


# Summary:
# - Dialog that buffers and displays reconstruction output.
# - What it does: owns a QTextEdit, trims output, and manages line buffering.
class VolumeReconstructionLogDialog(QDialog):
    # Summary:
    # - Initialize the dialog UI and output buffers.
    # - Input: `self`, `parent` (QWidget | None), `text_limit` (int).
    # - Returns: None.
    def __init__(
        self, parent: Optional[QWidget] = None, text_limit: int = RECON_LOG_TEXT_LIMIT
    ) -> None:
        super().__init__(parent)
        # Give the dialog a stable object name for consistency.
        self.setObjectName("volumeReconLogDialog")
        # UI state change: label the dialog so users know what it shows.
        self.setWindowTitle("Volume Reconstruction Output")
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.resize(600, 150)

        # Track the max character count to keep the log bounded.
        self._text_limit = text_limit
        # Track partial line buffers for stdout and stderr.
        self._stdout_line_buffer = ""
        self._stderr_line_buffer = ""
        # Track whether a run is actively streaming.
        self._is_running = False

        # Build the dialog layout with a read-only text view.
        layout = QVBoxLayout(self)
        self._text_edit = QTextEdit(self)
        self._text_edit.setObjectName("volumeReconLogTextEdit")
        # UI state change: keep the log view read-only.
        self._text_edit.setReadOnly(True)
        # UI state change: wrap long lines to avoid a horizontal scroll bar.
        self._text_edit.setLineWrapMode(QTextEdit.WidgetWidth)
        layout.addWidget(self._text_edit)

    # Summary:
    # - Start a new reconstruction run and reset the output view.
    # - Input: `self`.
    # - Returns: None.
    def start_new_run(self) -> None:
        # Track that a new run is active so close behavior is non-destructive.
        self._is_running = True
        # Reset buffers so output starts cleanly for the new run.
        self._reset_buffers()
        # UI state change: clear the visible log before showing the dialog.
        self._text_edit.clear()
        # UI state change: show the dialog and bring it forward.
        self.show()
        self.raise_()
        self.activateWindow()

    # Summary:
    # - Append a stdout chunk to the log with line buffering.
    # - Input: `self`, `text` (str).
    # - Returns: None.
    def append_stdout(self, text: str) -> None:
        # Stream stdout text into the shared output handler.
        self._append_output(text, is_stderr=False)

    # Summary:
    # - Append a stderr chunk to the log with line buffering and a prefix.
    # - Input: `self`, `text` (str).
    # - Returns: None.
    def append_stderr(self, text: str) -> None:
        # Stream stderr text into the shared output handler.
        self._append_output(text, is_stderr=True)

    # Summary:
    # - Mark the current reconstruction run as finished.
    # - Input: `self`.
    # - Returns: None.
    def finish_run(self) -> None:
        # Track that the run finished so close behavior can be relaxed.
        self._is_running = False
        # UI state change: close the dialog now that output is complete.
        self.close()

    # Summary:
    # - Hide the dialog without destroying it.
    # - Input: `self`.
    # - Returns: None.
    def close_dialog(self) -> None:
        # UI state change: hide the dialog so output can continue in the background.
        self.hide()

    # Summary:
    # - Reset the stdout and stderr partial line buffers.
    # - Input: `self`.
    # - Returns: None.
    def _reset_buffers(self) -> None:
        # Clear partial line buffers so output starts cleanly.
        self._stdout_line_buffer = ""
        self._stderr_line_buffer = ""

    # Summary:
    # - Append output text line-by-line with buffering for partial lines.
    # - Input: `self`, `text` (str), `is_stderr` (bool).
    # - Returns: None.
    def _append_output(self, text: str, is_stderr: bool) -> None:
        # Guard against empty chunks so we do not emit blank lines.
        if not text:
            return

        # Keep partial lines so we only append complete lines to the UI.
        pending = self._stderr_line_buffer if is_stderr else self._stdout_line_buffer
        combined = pending + text
        lines = combined.splitlines(keepends=True)
        new_buffer = ""

        # Append only complete lines to keep the log readable line-by-line.
        for line in lines:
            if line.endswith("\n") or line.endswith("\r"):
                clean_line = line.rstrip("\r\n")
                prefix = "[stderr] " if is_stderr else ""
                self._text_edit.append(f"{prefix}{clean_line}")
            else:
                new_buffer = line

        # Store any trailing partial line for the next read.
        if is_stderr:
            self._stderr_line_buffer = new_buffer
        else:
            self._stdout_line_buffer = new_buffer

        # UI state change: keep the log scrolled to the newest output.
        self._text_edit.moveCursor(QTextCursor.End)
        self._trim_log_text()

    # Summary:
    # - Trim the log text to a maximum character count.
    # - Input: `self`.
    # - Returns: None.
    def _trim_log_text(self) -> None:
        document = self._text_edit.document()
        current_count = document.characterCount()
        if current_count <= self._text_limit:
            return

        # Remove the oldest characters so the log does not grow unbounded.
        excess = current_count - self._text_limit
        cursor = QTextCursor(document)
        cursor.movePosition(QTextCursor.Start)
        cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, excess)
        cursor.removeSelectedText()

    # Summary:
    # - Handle the dialog close event without destroying the dialog while running.
    # - Input: `self`, `event` (QCloseEvent).
    # - Returns: None.
    def closeEvent(self, event) -> None:
        # Keep the dialog alive while a run is active so streaming can continue.
        if self._is_running:
            self.hide()
            event.ignore()
            return

        # Allow the default close behavior when no run is active.
        super().closeEvent(event)
