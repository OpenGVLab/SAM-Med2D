import sys
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton
import SimpleITK as sitk
import numpy as np

# Create a PyQt application and main window
app = QApplication(sys.argv)
main_window = QMainWindow()
main_window.setWindowTitle("3D NIfTI Viewer")
central_widget = QWidget()
main_window.setCentralWidget(central_widget)
layout = QVBoxLayout()
central_widget.setLayout(layout)

# Create a PyQtGraph OpenGLWidget to display the 3D image
view = gl.GLViewWidget()
layout.addWidget(view)

# Define a function to load and display the NIfTI image
def load_nifti_and_display():
    # Replace 'your_image.nii.gz' with the path to your NIfTI file
    nifti_file = "Demo 3d Data/BraTS2021_00000_0000.nii.gz"

    # Load the NIfTI image using SimpleITK
    sitk_image = sitk.ReadImage(nifti_file)

    # Convert the SimpleITK image to a NumPy array
    data = sitk.GetArrayFromImage(sitk_image)

    # Swap axes to match the expected shape by GLVolumeItem
    # data = np.swapaxes(data, 0, 2)  # Swap the first and third axes

    # Normalize the data to [0, 1]
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)

    # Create a volume item and add it to the view
    volume = gl.GLVolumeItem(normalized_data, sliceDensity=2, smooth=True)
    # volume.setLevels(min_val, max_val)  # Set levels for volume rendering
    view.addItem(volume)

# Create a button to trigger the loading and display of the NIfTI image
load_button = QPushButton("Load NIfTI Image")
load_button.clicked.connect(load_nifti_and_display)
layout.addWidget(load_button)

# Show the main window
main_window.show()
sys.exit(app.exec_())
