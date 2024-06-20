import numpy as np
from PIL import Image
import nibabel as nib
import SimpleITK as sitk
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider, QPushButton, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import Qt
from MainWindow import MainWindowSegment


class SlicerWidget(QWidget):
    def __init__(self, data_path, sitk_image):
        super().__init__()
        self.data_path = data_path
        self.nibabel_data = nib.load(self.data_path).get_fdata()
        self.sitk_image = sitk.GetArrayViewFromImage(sitk_image)
        self.current_view = 'sagittal'
        self.slice_index = max(self.nibabel_data.shape)// 2
        self.image_path = None
        self.slice_data_nibabel = None
        self.slice_data = None
        self.max_shape_size = max(self.nibabel_data.shape)
        self.sitk_image = self.pad_to_specific_shape(self.sitk_image, (
        max(self.nibabel_data.shape), max(self.nibabel_data.shape), max(self.nibabel_data.shape)))
        self.nibabel_data = self.pad_to_specific_shape(self.nibabel_data, (
        max(self.nibabel_data.shape), max(self.nibabel_data.shape), max(self.nibabel_data.shape)))
        print(self.sitk_image.shape)
        print(self.nibabel_data.shape)

    def pad_to_specific_shape(self, input_array, target_shape, pad_value=0):
        """
        Pad a NumPy array to a specific shape.

        Parameters:
            input_array (numpy.ndarray): The input array to be padded.
            target_shape (tuple): The desired shape (tuple of integers) of the padded array.
            pad_value (float or int, optional): The value used for padding. Default is 0.

        Returns:
            numpy.ndarray: The padded array with the specified shape.
        """
        # Ensure the input array and target shape have the same number of dimensions
        if len(input_array.shape) != len(target_shape):
            raise ValueError("Input array and target shape must have the same number of dimensions.")

        # Calculate the padding required for each dimension
        pad_width = [(0, max(0, target_shape[i] - input_array.shape[i])) for i in range(len(target_shape))]

        # Pad the input array
        padded_array = np.pad(input_array, pad_width, mode='constant', constant_values=pad_value)

        return padded_array

    def paintEvent(self, event):
        print(self.slice_index)
        print(self.current_view)
        print('---------------------------------------')
        painter = QPainter(self)

        if self.current_view == 'sagittal':
            self.slice_data = self.sitk_image[:, :, self.slice_index]
            self.slice_data_nibabel = self.nibabel_data[:, :, self.slice_index]
        if self.current_view == 'coronal':
            self.slice_data = self.sitk_image[:, self.slice_index, :]
            self.slice_data_nibabel = self.nibabel_data[:, self.slice_index, :]
        if self.current_view == 'axial':
            self.slice_data = self.sitk_image[self.slice_index, :, :]
            self.slice_data_nibabel = self.nibabel_data[self.slice_index, :, :]

        slice_data = ((self.slice_data - self.slice_data.min()) / (
                    self.slice_data.max() - self.slice_data.min()) * 255).astype('uint8')
        height, width = slice_data.shape
        bytes_per_line = width
        image = QImage(slice_data.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(image)
        painter.drawPixmap(0, 0, self.width(), self.height(), pixmap)

    def set_current_view(self, view):
        self.current_view = view
        self.slice_index = self.max_shape_size // 2
        self.update()

    def set_slice_index(self, index):
        self.slice_index = index
        self.update()

    def save_current_view_as_jpg(self):
        print(self.slice_index)
        print(self.current_view)
        print('********************************')
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getSaveFileName(self, f"Save {self.current_view.capitalize()} View as JPG", "",
                                                   "JPEG Image Files (*.jpg);;All Files (*)", options=options)
        rescaled = (255.0 / self.slice_data.max() * (
                    self.slice_data - self.slice_data.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        im.save(file_path)
        self.image_path = file_path


class MainWindow(QMainWindow):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.setWindowTitle("3D Slicer")
        self.setGeometry(100, 100, 800, 600)
        sitk_image = sitk.ReadImage(self.data_path)
        self.slicer_widget = SlicerWidget(self.data_path, sitk_image)

        self.scrollbar = QSlider(Qt.Horizontal)
        self.scrollbar.setMaximum(sitk_image.GetSize()[0] - 1)
        self.scrollbar.valueChanged.connect(self.slicer_widget.set_slice_index)

        self.save_button = QPushButton("Save as JPG")
        self.save_button.clicked.connect(self.slicer_widget.save_current_view_as_jpg)

        self.view_buttons = {
            'Sagittal': 'sagittal',
            'Coronal': 'coronal',
            'Axial': 'axial',
        }

        self.start_button = QPushButton("Start Segment")
        self.start_button.clicked.connect(self.close_window)

        for button_text, view in self.view_buttons.items():
            button = QPushButton(button_text)
            button.clicked.connect(lambda _, view=view: self.slicer_widget.set_current_view(view))
            self.view_buttons[button_text] = button

        layout = QVBoxLayout()
        layout.addWidget(self.slicer_widget)
        layout.addWidget(self.scrollbar)
        layout.addWidget(self.save_button)
        for button_text, button in self.view_buttons.items():
            layout.addWidget(button)
        layout.addWidget(self.start_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def close_window(self):
        self.close()

    def get_image_path(self):
        return self.slicer_widget.image_path
