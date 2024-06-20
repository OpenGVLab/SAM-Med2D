import os
import sys
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageQt
from PIL import Image

from Inference import Inference
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor


# Define the main application window class
class MainWindowSegment(QMainWindow):
    def __init__(self, image_path):
        """
        Initialize the main application window.
        The images at each section save at the 'Output' folder of the code base path with the name of 'iteration_n.png'.
        Parameters:
        - image_path (str): Path to the image to be displayed.
        """
        super().__init__()
        self.image_path = image_path
        self.main_image = cv2.imread(self.image_path)

        # Set up the main window
        self.setWindowTitle("SAM click base")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.image = QPixmap(image_path)
        self.image_size = (self.image.width(), self.image.height())
        self.image_screen_size = (500, 500)
        self.image = self.image.scaled(self.image_screen_size[0], self.image_screen_size[1])
        self.label = QLabel(self)
        self.label.setPixmap(self.image)
        self.label.setAlignment(Qt.AlignTop)
        self.resize(550, 300)
        self.label.mousePressEvent = self.handleMouseClick

        self.central_layout = QVBoxLayout()
        self.central_layout.addWidget(self.label)

        self.mask_btn = QPushButton("Mask")
        self.gt_btn = QPushButton("Ground Truth")
        self.predict_btn = QPushButton("Predict")
        self.undo_btn = QPushButton("Undo")

        self.mask_btn.clicked.connect(self.mask_btn_action)
        self.gt_btn.clicked.connect(self.gt_btn_action)
        self.predict_btn.clicked.connect(self.predict_btn_action)
        self.undo_btn.clicked.connect(self.undoCircleDraw)

        self.central_layout.addWidget(self.mask_btn)
        self.central_layout.addWidget(self.gt_btn)
        self.central_layout.addWidget(self.predict_btn)
        self.central_layout.addWidget(self.undo_btn)

        self.central_widget.setLayout(self.central_layout)

        self.model = Inference(self.image_path)
        self.point_flag = ''
        self.gt_points = []
        self.mask_points = []
        self.points_sequence = []
        self.iteration = 0
        self.all_images = [self.image]
        self.undo_counter = -2
        self.base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output')

    def mask_btn_action(self):
        """Set the point_flag to 'mask' when Mask button is clicked."""
        self.point_flag = 'mask'

    def gt_btn_action(self):
        """Set the point_flag to 'gt' when Ground Truth button is clicked."""
        self.point_flag = 'gt'

    def scale_points(self, points):
        """
        Scale the points to match the image dimensions.

        Parameters:
        - points (numpy.array): Array of points to be scaled.

        Returns:
        - List: Scaled points.
        """
        points[:, 0] = (points[:, 0] / self.image_screen_size[0]) * self.image_size[0]
        points[:, 1] = (points[:, 1] / self.image_screen_size[1]) * self.image_size[1]
        return points

    def draw_mask(self, image, mask_generated):
        """
        Draw a mask on the image.

        Parameters:
        - image (numpy.array): The original image.
        - mask_generated (numpy.array): The generated mask.

        Returns:
        - numpy.array: The image with the mask drawn.
        """
        masked_image = image.copy()

        mask_resized = cv2.resize(mask_generated, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        unique_labels = np.unique(mask_resized)

        for label in unique_labels:
            if label == 0:  # Skip background label
                continue

            color = np.random.randint(0, 100, size=(3,), dtype=np.uint8)

            label_mask = (mask_resized == label).astype(np.uint8)

            print(masked_image.shape)
            print(image.shape)
            masked_image[label_mask > 0] = color
        masked_image = masked_image.astype(np.uint8)
        return cv2.addWeighted(image, 0.1, masked_image, 0.9, 0)

    def save_image(self, base_directory, segmented_image, masks):
        current_datetime = datetime.now()
        timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        new_directory_path = os.path.join(base_directory, timestamp)
        if not os.path.exists(new_directory_path):
            os.makedirs(new_directory_path)

        fig, ax = plt.subplots()
        ax.set_axis_off()

        ax.imshow(segmented_image)
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = masks.shape[-2:]
        mask_image = masks.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        fig.savefig(os.path.join(new_directory_path, 'iteration_{}.png').format(self.iteration), bbox_inches='tight',
                    pad_inches=0)

    def predict_btn_action(self):
        """Handle the Predict button click event."""
        if len(self.gt_points) + len(self.mask_points) == 0:
            return None

        if len(self.gt_points) > 0:
            gt_np = np.array(self.gt_points)
        else:
            gt_np = np.empty((0, 2))  # Create an empty 2D array for gt_points

        if len(self.mask_points) > 0:
            mask_np = np.array(self.mask_points)
        else:
            mask_np = np.empty((0, 2))  # Create an empty 2D array for mask_points

        all_points = self.scale_points(np.concatenate((gt_np, mask_np)))
        all_labels = np.concatenate((np.zeros(len(self.gt_points)), np.ones(len(self.mask_points))))

        print(all_points)
        print(all_labels)
        if len(all_labels) != 0 and len(all_points) != 0:
            masks, scores, logits = self.model.creat_mask(all_points, all_labels)

        segmented_image = self.draw_mask(self.main_image, masks.squeeze())
        self.save_image(self.base_path, segmented_image, masks)

        print(segmented_image.shape)
        img = Image.fromarray(segmented_image, mode='RGB')
        qt_img = ImageQt.ImageQt(img)
        self.image = QPixmap.fromImage(qt_img)
        self.image = self.image.scaled(500, 500)

        self.label.setPixmap(self.image)
        self.label.setAlignment(Qt.AlignTop)

        self.iteration += 1  # Add the iteration number
        self.reset_undo_params()
        print('Predict process is complete!')
        print("----------------------------")

    def reset_undo_params(self):
        """
        # Clear the image list and reset the undo_coutner because the user can not undo points that have been used before
        # prediction process
        :return:
        """
        self.all_images = [self.image]
        self.undo_counter = -2

    def handleMouseClick(self, event):
        """Handle mouse click events on the image label."""
        if self.point_flag == 'gt':
            pos = event.pos()

            x = pos.x()
            y = pos.y()
            print(f"{self.point_flag} at ({x}, {y})")
            self.drawCircle(x, y)
            self.gt_points.append(np.array([x, y]))

        if self.point_flag == 'mask':
            pos = event.pos()

            x = pos.x()
            y = pos.y()
            print(f"{self.point_flag} at ({x}, {y})")
            self.drawCircle(x, y)
            self.mask_points.append(np.array([x, y]))

    def drawCircle(self, x, y):
        """Draw a circle on the image at the specified position (x, y)."""
        self.image = QPixmap(self.image)
        painter = QPainter(self.image)
        if self.point_flag == 'gt':
            painter.setPen(QPen(Qt.red, 5))
        elif self.point_flag == 'mask':
            painter.setPen(QPen(Qt.green, 5))
        painter.drawEllipse(x, y, 1, 1)
        painter.end()
        self.label.setPixmap(self.image)
        self.label.setAlignment(Qt.AlignTop)
        self.last_x_y = (x, y, self.point_flag)
        self.all_images.append(self.image)

    def undoCircleDraw(self):
        """Undo the last circle draw action."""
        if abs(self.undo_counter + 2) > len(self.all_images):
            return None
        self.image = self.all_images[self.undo_counter]
        self.label.setPixmap(self.image)
        self.label.setAlignment(Qt.AlignTop)

        if self.last_x_y[0] == 'gt':
            self.gt_points.pop()
        elif self.last_x_y[0] == 'mask':
            self.mask_points.pop()
        self.undo_counter -= 1  # The counter of the undo for when the use give the button of the undo for many times.


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindowSegment(image_path='/home/mkhanmhmdi/Downloads/SAM(click base)/SAM-Med2D/PyQt/a.png')
    window.show()
    sys.exit(app.exec_())
