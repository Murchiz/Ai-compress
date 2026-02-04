import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QProgressBar, QTextEdit, QListWidget, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from backend.core.manager import BrainManager

class WorkerThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, action, input_path, output_path, manager):
        super().__init__()
        self.action = action
        self.input_path = input_path
        self.output_path = output_path
        self.manager = manager

    def run(self):
        try:
            if self.action == "compress":
                # Redirect stdout to capture the prints if needed, or just emit signals
                res = self.manager.compress_and_learn(self.input_path, self.output_path)
                self.finished.emit(res)
            else:
                self.manager.decompress(self.input_path, self.output_path)
                self.finished.emit("Decompression successful")
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.manager = BrainManager()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("AI Continuous Learning Compressor")
        self.setGeometry(100, 100, 600, 500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Brain Status
        status_layout = QHBoxLayout()
        self.brain_label = QLabel(f"Current Brain: {self.manager.get_current_brain_id()}")
        status_layout.addWidget(self.brain_label)
        layout.addLayout(status_layout)

        # File List / Selection
        layout.addWidget(QLabel("Files to Process (First file will be processed):"))
        self.file_list = QListWidget()
        layout.addWidget(self.file_list)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add File")
        add_btn.clicked.connect(self.add_file)
        btn_layout.addWidget(add_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.file_list.clear)
        btn_layout.addWidget(clear_btn)
        layout.addLayout(btn_layout)

        # Actions
        action_layout = QHBoxLayout()
        self.compress_btn = QPushButton("Compress & Learn")
        self.compress_btn.clicked.connect(self.start_compress)
        action_layout.addWidget(self.compress_btn)

        self.decompress_btn = QPushButton("Decompress")
        self.decompress_btn.clicked.connect(self.start_decompress)
        action_layout.addWidget(self.decompress_btn)
        layout.addLayout(action_layout)

        # Progress & Log
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

    def add_file(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files")
        if files:
            self.file_list.addItems(files)

    def log(self, message):
        self.log_output.append(message)

    def start_compress(self):
        items = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not items:
            QMessageBox.warning(self, "Error", "Please add a file first.")
            return

        # For simplicity, we compress the first file in the list
        input_path = items[0]
        output_path, _ = QFileDialog.getSaveFileName(self, "Save Compressed File", "", "AICP Files (*.aicp)")

        if not output_path:
            return

        self.set_ui_enabled(False)
        self.log(f"Starting compression of {input_path}...")
        self.progress_bar.setRange(0, 0) # Indeterminate

        self.thread = WorkerThread("compress", input_path, output_path, self.manager)
        self.thread.finished.connect(self.on_compress_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def start_decompress(self):
        items = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        # Allow selecting from dialog even if list is empty
        input_path = items[0] if items else ""
        if not input_path or not input_path.endswith(".aicp"):
             input_path, _ = QFileDialog.getOpenFileName(self, "Select AICP File", "", "AICP Files (*.aicp)")

        if not input_path:
            return

        output_path, _ = QFileDialog.getSaveFileName(self, "Save Decompressed File")
        if not output_path:
            return

        self.set_ui_enabled(False)
        self.log(f"Starting decompression of {input_path}...")
        self.progress_bar.setRange(0, 0)

        self.thread = WorkerThread("decompress", input_path, output_path, self.manager)
        self.thread.finished.connect(self.on_decompress_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_compress_finished(self, new_brain_id):
        self.set_ui_enabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.log(f"Compression finished! New Brain ID: {new_brain_id}")
        self.brain_label.setText(f"Current Brain: {new_brain_id}")
        QMessageBox.information(self, "Success", f"File compressed. Model updated to {new_brain_id}")

    def on_decompress_finished(self, message):
        self.set_ui_enabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.log(message)
        QMessageBox.information(self, "Success", message)

    def on_error(self, error_msg):
        self.set_ui_enabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.log(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)

    def set_ui_enabled(self, enabled):
        self.compress_btn.setEnabled(enabled)
        self.decompress_btn.setEnabled(enabled)
        self.file_list.setEnabled(enabled)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
