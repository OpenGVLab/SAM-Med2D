import sys
import SimpleITK as sitk
from PyQt5.QtWidgets import QApplication
from BasePyQT import MainWindow
from MainWindow import MainWindowSegment


if __name__ == '__main__':
    sitk_image = sitk.ReadImage('/home/mkhanmhmdi/Downloads/SAM(click base)/SAM-Med2D/PyQt/Demo 3d '
                                'Data/BraTS2021_00000_0000.nii.gz')
    app1 = QApplication(sys.argv)
    window = MainWindow(sitk_image)
    window.show()
    exit_app1 = (app1.exec_())
    app = QApplication(sys.argv)
    window = MainWindowSegment()
    window.show()
    sys.exit(app.exec_())