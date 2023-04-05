import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

stride=32
auto=True
conf_thres=0.25
iou_thres=0.45
num_classes = 80
agnostic_nms = False
device = select_device("cpu")

import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',device=device,force_reload=True) 
#model = DetectMultiBackend("/mnt/disk1/object-detection/Santosh/yolov5/runs/train/exp8/weights/best.pt", device=device, dnn=False)

def main(img):
    cropped_image = None

    try:
        im0 = img.copy()

        results = model(img)

        pred = results.pandas().xyxy[0]

        index_1_pred = pred.loc[0, :].values.tolist()

        print(index_1_pred,"index_1_pred")
        label = index_1_pred[6]
        bbox = index_1_pred[:-3]
        return bbox
        if label== "person":
            print(bbox)
            cropped_image = im0[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]


    except Exception as e:
        error =  getattr(e, 'message', repr(e)) 
        print(error)

    return cropped_image
                 
if __name__=="__main__":
    img_folder = "filtered/"
    outfolder = img_folder[:-1]+"cropped/"
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    for file in os.listdir(img_folder):
        img_path = img_folder+file
        img = cv2.imread(img_path)
        print(img_path,img.shape)
        cropped_image = main(img)
        outpath  = outfolder+file
        if cropped_image is not None:
            cv2.imwrite(outpath,cropped_image)
