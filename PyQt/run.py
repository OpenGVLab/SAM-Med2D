import sys
import argparse
import SimpleITK as sitk
from PyQt5.QtWidgets import QApplication
from BasePyQT import MainWindow
from MainWindow import MainWindowSegment


def parse_arguments():
    parser = argparse.ArgumentParser(description="Segmentation Tool")
    parser.add_argument("data_path", type=str, help="Path to the data file (e.g., .nii.gz)")
    return parser.parse_args()


def run(data_path):
    app1 = QApplication(sys.argv)
    window = MainWindow(data_path)
    window.show()
    exit_app1 = (app1.exec_())
    image_path = window.get_image_path()

    app = QApplication(sys.argv)
    window = MainWindowSegment(image_path=image_path)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    args = parse_arguments()
    run(args.data_path)
