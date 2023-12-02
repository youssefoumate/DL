import argparse
import cv2
import torch
from maskrcnn_export import get_sample_inputs, setup_cfg

def inference(cfg):
    with torch.no_grad():
        model = torch.jit.load('output/model.ts')
        model.eval()
        input = get_sample_inputs(cfg)
        input = input[0]["image"]
        image_vis = torch.permute(input, (1, 2, 0)).numpy()
        output = model(input)[0].numpy()
        for box in zip(output):
            box = box[0].astype(int)
            image_vis = cv2.rectangle(image_vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 1)
        cv2.imwrite("output.jpg", image_vis)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a model for deployment.")
    parser.add_argument(
        "--format",
        choices=["caffe2", "onnx", "torchscript"],
        help="output format",
        default="torchscript",
    )
    parser.add_argument(
        "--export-method",
        choices=["caffe2_tracing", "tracing", "scripting"],
        help="Method to export models",
        default="tracing",
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--sample-image", default=None, type=str, help="sample image for input")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", help="output directory for the converted model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg = setup_cfg(args)
    inference(cfg)