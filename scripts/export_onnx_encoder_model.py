import sys
sys.path.append('.')

import os
import cv2
import onnx
import pathlib
import shutil
import argparse
import warnings
import numpy as np
import albumentations as A

import torch
import torch.nn as nn
from torch.nn import functional as F
from tempfile import mkdtemp
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
from onnx.external_data_helper import convert_model_to_external_data

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False


parser = argparse.ArgumentParser(
    description="Export the model to ONNX format with encoder support."
)

parser.add_argument(
    "--sam_checkpoint",
    type=str,
    required=True,
    help="The path to the SAM-Med2D model checkpoint.",
)

parser.add_argument(
    "--output", type=str, required=True, help="The filename to save the ONNX model to."
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="In ['default', 'vit_h', 'vit_l', 'vit_b']. Which type of SAM-Med2D model to export.",
)

parser.add_argument(
    "--use-preprocess",
    action="store_true",
    help="Integrate preprocessing into the model.",
)

parser.add_argument(
    "--opset",
    type=int,
    default=17,
    help="The ONNX opset version to use. Must be >=11",
)

parser.add_argument(
    '--device', 
    type=str, 
    default='cpu'
)

parser.add_argument(
    "--image_size", 
    type=int, 
    default=256, 
    help="image_size"

)

parser.add_argument(
    "--encoder_adapter", 
    type=bool, 
    default=True, 
    help="use adapter"
)

parser.add_argument(
    "--quantize-out",
    type=str,
    default=None,
    help=(
        "If set, will quantize the model and save it with this name. "
        "Quantization is performed with quantize_dynamic from onnxruntime.quantization.quantize."
    ),
)

parser.add_argument(
    "--gelu-approximate",
    action="store_true",
    help=(
        "Replace GELU operations with approximations using tanh. Useful "
        "for some runtimes that have slow or unimplemented erf ops, used in GELU."
    ),
)

class OnnxEncoderModel(nn.Module):

    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std  = [58.395, 57.12, 57.375]

    def __init__(
        self,
        model: Sam,
        input_size: tuple = (256, 256),
        pixel_mean: list = [123.675, 116.28, 103.53],
        pixel_std: list=[58.395, 57.12, 57.375],
        use_preprocess: bool = False
    ):
        super().__init__()
        self.use_preprocess = use_preprocess
        self.pixel_mean = torch.tensor(pixel_mean, dtype=torch.float)
        self.pixel_std = torch.tensor(pixel_std, dtype=torch.float)
        self.input_size = input_size
        self.model = model
        self.image_encoder = model.image_encoder
    
    @torch.no_grad()
    def forward(self, input_image: torch.Tensor):
        if self.use_preprocess:
            input_image = self.preprocess(input_image)
        image_embeddings = self.image_encoder(input_image)
        return image_embeddings

    def preprocess(self, input_image: torch.Tensor) -> torch.Tensor:
        """Image transform

        This function can convert the input image to the required input format for VIT.

        Args:
            img (torch.Tensor): Input image in BGR format.

        Returns:
            torch.Tensor: Transformed image.
        """

        # Normalization
        input_image = (input_image - self.pixel_mean) / self.pixel_std

        # permute channels
        input_image = torch.permute(input_image, (2, 0, 1))

        # CHW -> NCHW & Resize
        input_image = F.interpolate(input_image.unsqueeze(0), size=self.input_size, mode='nearest')

        return input_image


def run_export(args):
    print("Loading model...")
    sam = sam_model_registry[args.model_type](args).to(args.device)

    model = OnnxEncoderModel(
        model=sam,
        use_preprocess=args.use_preprocess,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if args.gelu_approximate:
        for _, m in model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    image_size = sam.image_encoder.img_size
    if args.use_preprocess:
        dummy_inputs = {
            "input_image": torch.randn(
                (image_size, image_size, 3), dtype=torch.float
            )
        }
        dynamic_axes = {
            "input_image": {0: "image_height", 1: "image_width"},
        }
    else:
        dummy_inputs = {
            "input_image": torch.randn(
                (1, 3, image_size, image_size), dtype=torch.float
            )
        }
        dynamic_axes = None
    _ = model(**dummy_inputs)

    output_names = ["image_embeddings"]
    onnx_base = os.path.splitext(os.path.basename(args.output))[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Exporting onnx model to {args.output}...")
        if args.model_type == "vit_h":
            tmp_dir = mkdtemp()
            tmp_model_path = os.path.join(tmp_dir, f"{onnx_base}.onnx")
            torch.onnx.export(
                model,
                tuple(dummy_inputs.values()),
                tmp_model_path,
                export_params=True,
                verbose=False,
                opset_version=args.opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

            # Combine the weights into a single file
            pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            model = onnx.load(tmp_model_path)
            convert_model_to_external_data(
                model,
                all_tensors_to_one_file=True,
                location=f"{onnx_base}_data.bin",
                size_threshold=1024,
                convert_attribute=False,
            )

            # Save the model
            onnx.save(model, args.output)

            # Cleanup the temporary directory
            shutil.rmtree(tmp_dir)
        else:
            with open(args.output, "wb") as f:
                torch.onnx.export(
                    model,
                    tuple(dummy_inputs.values()),
                    f,
                    export_params=True,
                    verbose=False,
                    opset_version=args.opset,
                    do_constant_folding=True,
                    input_names=list(dummy_inputs.keys()),
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )

    if onnxruntime_exists:
        ort_inputs = {k: to_numpy(v) for k, v in dummy_inputs.items()}
        ort_session = onnxruntime.InferenceSession(args.output)
        _ = ort_session.run(None, ort_inputs)
        print("Model has successfully been run with ONNXRuntime.")


def to_numpy(tensor):
    return tensor.cpu().numpy()


if __name__ == "__main__":
    args = parser.parse_args()
    run_export(args=args)

    if args.quantize_out is not None:
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

        print(f"Quantizing model and writing to {args.quantize_out}...")
        quantize_dynamic(
            model_input=args.output,
            model_output=args.quantize_out,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        print("Done!")
