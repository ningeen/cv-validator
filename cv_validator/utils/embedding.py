from pathlib import Path

import cv2
import numpy as np

supported_models = {
    "efficientnet-lite4": "https://github.com/onnx/models/blob/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx?raw=true",
    "efficientnet-lite4-int8": "https://github.com/onnx/models/blob/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11-int8.onnx?raw=true",
}


def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def resize_with_aspect_ratio(
    img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR
):
    height, width, _ = img.shape
    new_height = int(100.0 * out_height / scale)
    new_width = int(100.0 * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img


def pre_process_edge_tpu(img, dims=(224, 224, 3)):
    output_height, output_width, _ = dims
    img = resize_with_aspect_ratio(
        img, output_height, output_width, inter_pol=cv2.INTER_LINEAR
    )
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype="float32")
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img
