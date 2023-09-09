import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Any, Union
from copy import deepcopy


parser = argparse.ArgumentParser(
    description="Inference an image with onnxruntime backend."
)

parser.add_argument(
    "--encoder_model",
    type=str,
    required=True,
    help="Path to the SAM-Med2D onnx encoder model.",
)

parser.add_argument(
    "--decoder_model",
    type=str,
    required=True,
    help="Path to the SAM-Med2D onnx decoder model.",
)

parser.add_argument(
    "--img_path",
    type=str,
    default="../../data_demo/images/amos_0507_31.png",
    help="Path to the image",
)

parser.add_argument(
    "--input_size", 
    type=int, 
    default=256, 
    help="input_size"
)

parser.add_argument(
    "--work_dir", 
    type=str, 
    default="workdir", 
    help="work dir"
)

args = parser.parse_args()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   


class SamEncoder:
    """Sam encoder model.

    In this class, encoder model will encoder the input image.

    Args:
        model_path (str): sam encoder onnx model path.
        device (str): Inference device, user can choose 'cuda' or 'cpu'. default to 'cuda'.
        warmup_epoch (int): Warmup, if set 0,the model won`t use random inputs to warmup. default to 3.
    """

    def __init__(self,
                 model_path: str,
                 device: str = "cuda",
                 warmup_epoch: int = 3,
                 **kwargs):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ['CUDAExecutionProvider']
        elif device == "cpu":
            provider = ['CPUExecutionProvider']
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        print("loading encoder model...")
        self.session = ort.InferenceSession(model_path,
                                            opt,
                                            providers=provider,
                                            **kwargs)

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape

        self.pixel_mean = np.array([123.675, 116.28, 103.53])
        self.pixel_std = np.array([58.395, 57.12, 57.375])
        self.input_size = (self.input_shape[-1], self.input_shape[-2])

        if warmup_epoch:
            self.warmup(warmup_epoch)

    def warmup(self, epoch: int) -> None:
        """warmup function

        Args:
            epoch (int): warmup epoch.
        """
        x = np.random.random(self.input_shape).astype(np.float32)
        print("start warmup!")
        for i in tqdm(range(epoch)):
            self.session.run(None, {self.input_name: x})
        print("warmup finish!")

    def transform(self, img: np.ndarray) -> np.ndarray:
        """image transform

        This function can convert the input image to the required input format for vit.

        Args:
            img (np.ndarray): input image, the image type should be BGR.

        Returns:
            np.ndarray: transformed image.
        """
        # BGR -> RGB
        input_image = img[..., ::-1]

        # Normalization
        input_image = (input_image - self.pixel_mean) / self.pixel_std

        # Resize
        input_image = cv2.resize(input_image, self.input_size, cv2.INTER_NEAREST)

        # HWC -> CHW
        input_image = input_image.transpose((2, 0, 1))

        # CHW -> NCHW
        input_image = np.expand_dims(input_image, 0).astype(np.float32)

        return input_image

    def _extract_feature(self, tensor: np.ndarray) -> np.ndarray:
        """extract image feature

        this function can use vit to extract feature from transformed image.

        Args:
            tensor (np.ndarray): input image with BGR format.

        Returns:
            np.ndarray: image`s feature.
        """
        input_image = self.transform(tensor)
        assert list(input_image.shape) == self.input_shape
        feature = self.session.run(None, {self.input_name: input_image})[0]
        assert list(feature.shape) == self.output_shape
        return feature

    def __call__(self, img: np.array, *args: Any, **kwds: Any) -> Any:
        return self._extract_feature(img)

class SamDecoder:
    """Sam decoder model.

    This class is the sam prompt encoder and lightweight mask decoder.

    Args:
        model_path (str): decoder model path.
        device (str): Inference device, user can choose 'cuda' or 'cpu'. default to 'cuda'.
    """

    def __init__(self,
                 model_path: str,
                 device: str = "cuda",
                 img_size: int = 256,
                 **kwargs):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ['CUDAExecutionProvider']
        elif device == "cpu":
            provider = ['CPUExecutionProvider']
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        print("loading decoder model...")
        self.mask_threshold = 0.5
        self.img_size = (img_size, img_size)
        self.session = ort.InferenceSession(model_path,
                                            opt,
                                            providers=provider,
                                            **kwargs)

    def run(self,
            img_embeddings: np.ndarray,
            origin_image_size: Union[list, tuple],
            point_coords: Union[list, np.ndarray] = None,
            point_labels: Union[list, np.ndarray] = None,
            boxes: Union[list, np.ndarray] = None,
            mask_input: np.ndarray = None,
            return_logits: bool = False):
        """decoder forward function

        This function can use image feature and prompt to generate mask. Must input
        at least one box or point.

        Args:
            img_embeddings (np.ndarray): the image feature from vit encoder.
            origin_image_size (list or tuple): the input image size.
            point_coords (list or np.ndarray): the input points.
            point_labels (list or np.ndarray): the input points label, 1 indicates
                a foreground point and 0 indicates a background point.
            boxes (list or np.ndarray): A length 4 array given a box prompt to the
                model, in XYXY format.
            mask_input (np.ndarray): A low resolution mask input to the model,
                typically coming from a previous prediction iteration. Has form
                1xHxW, where for SAM, H=W=4 * embedding.size.

        Returns:
            the segment results.
        """
        if point_coords is None and point_labels is None and boxes is None:
            raise ValueError("Unable to segment, please input at least one box or point.")

        if img_embeddings.shape != (1, 256, 16, 16):
            raise ValueError("Got wrong embedding shape!")
        if mask_input is None:
            mask_input = np.zeros((1, 1, 64, 64), dtype=np.float32)
            has_mask_input = np.zeros(1, dtype=np.float32)
        else:
            mask_input = np.expand_dims(mask_input, axis=0)
            has_mask_input = np.ones(1, dtype=np.float32)
            if mask_input.shape != (1, 1, 64, 64):
                raise ValueError("Got wrong mask!")
        if point_coords is not None:
            if isinstance(point_coords, list):
                point_coords = np.array(point_coords, dtype=np.float32)
            if isinstance(point_labels, list):
                point_labels = np.array(point_labels, dtype=np.float32)

        if point_coords is not None:
            point_coords = self.apply_coords(point_coords, origin_image_size, self.img_size).astype(np.float32)
            point_coords = np.expand_dims(point_coords, axis=0)
            point_labels = np.expand_dims(point_labels, axis=0)

        if boxes is not None:
            if isinstance(boxes, list):
                boxes = np.array(boxes, dtype=np.float32)
            assert boxes.shape[-1] == 4

            boxes = self.apply_boxes(boxes, origin_image_size, self.img_size).reshape((1, -1, 2)).astype(np.float32)
            box_label = np.array([[2, 3] for i in range(boxes.shape[1] // 2)], dtype=np.float32).reshape((1, -1))

            if point_coords is not None:
                point_coords = np.concatenate([point_coords, boxes], axis=1)
                point_labels = np.concatenate([point_labels, box_label], axis=1)
            else:
                point_coords = boxes
                point_labels = box_label

        assert point_coords.shape[0] == 1 and point_coords.shape[-1] == 2
        assert point_labels.shape[0] == 1
        print(f"point_coords={point_coords}, point_labels={point_labels}")
        input_dict = {"image_embeddings": img_embeddings,
                      "point_coords": point_coords,
                      "point_labels": point_labels,
                      "mask_input": mask_input,
                      "has_mask_input": has_mask_input,
                      "orig_im_size": np.array(origin_image_size, dtype=np.float32)}
        masks, iou_predictions, low_res_masks = self.session.run(None, input_dict)

        if not return_logits:
            sigmoid_output = self.sigmoid(masks)
            masks = (sigmoid_output > self.mask_threshold).astype(np.float32)

        return masks[0], iou_predictions[0], low_res_masks[0]

    @staticmethod
    def sigmoid(x):
        return 0.5 * (np.tanh(0.5 * x) + 1)

    def apply_coords(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes.reshape(-1, 4)

def main():
    # Create save folder
    save_path = os.path.join(args.work_dir, 'ort_demo_results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    base_name, file_extension = os.path.splitext(os.path.basename(args.img_path))

    # Initialize the SAM-Med2D onnx model
    encoder = SamEncoder(
        model_path=args.encoder_model,
        warmup_epoch=3
    )
    decoder = SamDecoder(
        model_path=args.decoder_model,
    )

    '''Specifying a specific object with a point'''
    img_file = cv2.imread(args.img_path)
    img_embeddings = encoder(img_file)
    
    origin_image_size = img_file.shape[:2]
    point_coords = np.array([[162, 127]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.float32)
    masks, _, logits = decoder.run(
        img_embeddings=img_embeddings,
        origin_image_size=origin_image_size,
        point_coords=point_coords,
        point_labels=point_labels
    )
    
    plt.figure(figsize=(10,10))
    plt.imshow(img_file)
    show_mask(masks, plt.gca())
    show_points(point_coords, point_labels, plt.gca())
    plt.axis('off')
    plt.savefig(os.path.join(save_path, base_name+'_point1'+file_extension))
    plt.show()  

    '''Optimizing Segmentation Results by Point Interaction'''
    new_point_coords = np.array([[169, 140]], dtype=np.float32)
    new_point_labels = np.array([0], dtype=np.float32)
    point_coords = np.concatenate((point_coords, new_point_coords))
    point_labels = np.concatenate((point_labels, new_point_labels))
    mask_inputs = 1. / (1. + np.exp(-logits.astype(np.float32)))

    masks, _, logits = decoder.run(
        img_embeddings=img_embeddings,
        origin_image_size=origin_image_size,
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input = mask_inputs,
    )

    plt.figure(figsize=(10,10))
    plt.imshow(img_file)
    show_mask(masks, plt.gca())
    show_points(point_coords, point_labels, plt.gca())
    plt.axis('off')
    plt.savefig(os.path.join(save_path, base_name+'_point2'+file_extension))
    plt.show()

    '''Specifying a specific object with a bounding box'''
    boxes = np.array([135,100,180,150])

    masks, _, _ = decoder.run(
        img_embeddings=img_embeddings,
        origin_image_size=origin_image_size,
        boxes=boxes,
    )
    plt.figure(figsize=(10,10))
    plt.imshow(img_file)
    show_mask(masks, plt.gca())
    show_box(boxes, plt.gca())
    plt.axis('off')
    plt.savefig(os.path.join(save_path, base_name+'_box'+file_extension))
    plt.show()  
 

if __name__ == '__main__':
    main()