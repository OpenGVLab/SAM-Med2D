import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor
from argparse import Namespace

class QTModel:
    def __init__(self):
        self.args = Namespace()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.image_size = 256
        self.args.encoder_adapter = True
        self.args.sam_checkpoint = "/home/hooshman/Documents/reza_javadzade/Interactive/SAM-Med2D/Pretrain-Models/sam-med2d_b.pth"
        self.model = None
        self.predictor = None
        self.load_model()
        self.image = cv2.imread('/home/hooshman/Documents/reza_javadzade/Interactive/SAM-Med2D/data_demo/Brats/0.png')
        self.set_image()

    def load_model(self):
        self.model = sam_model_registry["vit_b"](self.args).to(self.device)
        self.predictor = SammedPredictor(self.model)

    def set_image(self):
        self.predictor.set_image(self.image)

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=100):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white',
                   linewidth=0.5)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white',
                   linewidth=0.5)

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def creat_mask(self, points, labels):
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        return masks, scores, logits