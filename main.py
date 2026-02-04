import sys
import os

# Add the current directory to sys.path so we can import backend
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frontend.app import MainWindow
from PyQt6.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
