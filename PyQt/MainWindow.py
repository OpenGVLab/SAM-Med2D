import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageQt

from Inference import Inference
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QGraphicsView, QGraphicsScene, QPushButton, \
    QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QImage


class MainWindowSegment(QMainWindow):
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.main_image = cv2.imread(self.image_path)

        self.setWindowTitle("SAM click base")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.image = QPixmap(image_path)
        self.label = QLabel(self)
        self.label.setPixmap(self.image)
        self.label.setAlignment(Qt.AlignTop)

        # self.setCentralWidget(self.label)
        # self.setWindowTitle('Image Viewer')
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
        self.last_x_y = None
        self.iteration = 0

    def mask_btn_action(self):
        self.point_flag = 'mask'

    def gt_btn_action(self):
        self.point_flag = 'gt'

    def draw_mask(self, image, mask_generated):
        masked_image = image.copy()

        # Resize the mask to match the dimensions of the image
        mask_resized = cv2.resize(mask_generated, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Find unique labels in the resized mask
        unique_labels = np.unique(mask_resized)

        # Iterate through each unique label and assign a unique color
        for label in unique_labels:
            if label == 0:  # Skip background label
                continue

            # Generate a random color for each label
            color = np.random.randint(0, 100, size=(3,), dtype=np.uint8)

            # Create a binary mask for the current label
            label_mask = (mask_resized == label).astype(np.uint8)

            print('hi')
            print(masked_image.shape)
            print(image.shape)
            # Set the color for the pixels belonging to the current label
            masked_image[label_mask > 0] = color

        masked_image = masked_image.astype(np.uint8)

        # You can adjust the alpha and beta values to control the blending
        return cv2.addWeighted(image, 0.1, masked_image, 0.9, 0)

    def predict_btn_action(self):
        if len(self.gt_points) > 0 or len(self.mask_points) > 0:
            gt_np = np.array(self.gt_points)
            mask_np = np.array(self.mask_points)
            all_points = np.concatenate((gt_np, mask_np))
            all_labels = np.concatenate((np.zeros(len(self.gt_points)), np.ones(len(self.mask_points))))

        print(all_points)
        print(all_labels)
        if len(all_labels) != 0 and len(all_points) != 0:
            masks, scores, logits = self.model.creat_mask(all_points, all_labels)

        segmented_image = self.draw_mask(self.main_image, masks.squeeze())

        fig, ax = plt.subplots()
        ax.set_axis_off()

        # ax.imshow(self.main_image)
        ax.imshow(segmented_image)
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = masks.shape[-2:]
        mask_image = masks.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        fig.savefig('iteration_{}.png'.format(self.iteration), bbox_inches='tight', pad_inches=0)

        # self.image = QPixmap("iteration_{}.png".format(self.iteration))
        from PIL import Image
        from matplotlib import cm
        print("-----------")
        print(segmented_image.shape)
        img = Image.fromarray(segmented_image, mode='RGB')
        qt_img = ImageQt.ImageQt(img)
        self.image = QPixmap.fromImage(qt_img)
        self.image = self.image.scaled(500,500)

        self.label.setPixmap(self.image)
        self.label.setAlignment(Qt.AlignTop)
        # self.resize(400, 300)

        self.iteration += 1  # Add the iteration number
        # mask = self.QPixmap(masks)
        # result_pixmap = self.join_pixmap(self.photo.pixmap(), self.label.pixmap())

        print('get the predict')

    def handleMouseClick(self, event):
        if self.point_flag == 'gt':
            pos = event.pos()
            # Calculate the position of the mouse cursor relative to the QPixmap
            x = pos.x()
            y = pos.y()
            print(f"{self.point_flag} at ({x}, {y})")
            self.drawCircle(x, y)
            self.gt_points.append(np.array([x, y]))

        if self.point_flag == 'mask':
            pos = event.pos()

            # Calculate the position of the mouse cursor relative to the QPixmap
            x = pos.x()
            y = pos.y()
            print(f"{self.point_flag} at ({x}, {y})")
            self.drawCircle(x, y)
            self.mask_points.append(np.array([x, y]))

    def drawCircle(self, x, y):
        # Create a new pixmap with the same size as the original image
        self.image = QPixmap(self.image)
        # self.label.setPixmap(new_pixmap)

        # Initialize a QPainter object with the new pixmap
        painter = QPainter(self.image)

        # Set the pen color and width
        if self.point_flag == 'gt':
            painter.setPen(QPen(Qt.red, 5))
        elif self.point_flag == 'mask':
            painter.setPen(QPen(Qt.green, 5))

        # Draw the circle at the specified position
        painter.drawEllipse(x, y, 1, 1)

        # End the painting session
        painter.end()

        # Show the new pixmap in the label
        self.label.setPixmap(self.image)
        self.label.setAlignment(Qt.AlignTop)

        # Save the last x and y position click.
        self.last_x_y = (x, y, self.point_flag)

    def undoCircleDraw(self):
        # Create a new pixmap with the same size as the original image
        self.image = QPixmap(self.image)
        # self.label.setPixmap(new_pixmap)
        painter = QPainter(self.image)
        painter.fillRect(self.last_x_y[0], self.last_x_y[1], 5, 5, QColor(0, 0, 100))
        # End the painting session
        painter.end()

        # Show the new pixmap in the label
        self.label.setPixmap(self.image)
        self.label.setAlignment(Qt.AlignTop)

        if self.last_x_y[3] == 'gt':
            self.gt_points.pop()
        elif self.last_x_y[3] == 'mask':
            self.mask_points.pop()

    def updateMask(self, mask):
        pixmap = QPixmap(self.image, QImage.Format_RGB32)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.drawImage(0, 0, self.image)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.drawImage(0, 0, mask)
        painter.end()
        # Show the new pixmap in the label
        self.label.setPixmap(self.image)
        self.label.setPixmap(pixmap)
        self.label.setAlignment(Qt.AlignTop)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindowSegment(image_path='/home/mkhanmhmdi/Downloads/SAM(click base)/SAM-Med2D/PyQt/mhy.png')
    window.show()
    sys.exit(app.exec_())
