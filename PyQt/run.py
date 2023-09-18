import sys
import SimpleITK as sitk
from PyQt5.QtWidgets import QApplication
from BasePyQT import MainWindow
from MainWindow import MainWindowSegment


def control_windows():
    sitk_image = sitk.ReadImage('/home/mkhanmhmdi/Downloads/SAM(click base)/SAM-Med2D/PyQt/Demo 3d '
                                'Data/BraTS2021_00000_0000.nii.gz')
    app1 = QApplication(sys.argv)
    window = MainWindow(sitk_image)
    window.show()
    exit_app1 = (app1.exec_())


if __name__ == '__main__':
    data_path = '/home/mkhanmhmdi/Downloads/SAM(click base)/SAM-Med2D/PyQt/Demo 3d Data/BraTS2021_00000_0000.nii.gz'
    # while True:
    app1 = QApplication(sys.argv)
    window = MainWindow(data_path)
    window.show()
    exit_app1 = (app1.exec_())
    image_path = window.get_image_path()

    app = QApplication(sys.argv)
    window = MainWindowSegment(image_path=image_path)
    window.show()
    sys.exit(app.exec_())
