from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QPushButton, QWidget

class MainWindowSegment(QMainWindow):
    undo_signal = pyqtSignal()  # Custom signal to indicate "Undo" button click

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM click base")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.Mask = QPushButton("Mask")
        self.GT = QPushButton("Grand Truth")
        self.predict = QPushButton("Predict")
        self.undo = QPushButton("Undo")
        self.undo.clicked.connect(self.undo_btn)

        layout.addWidget(self.Mask)
        layout.addWidget(self.GT)
        layout.addWidget(self.predict)
        layout.addWidget(self.undo)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def undo_btn(self):
        self.close()
