import sys
import os
import SimpleITK as sitk
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider, QPushButton, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import Qt


class MainWindowSegment(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM click base")
        self.setGeometry(100, 100, 800, 600)
