import os
import argparse
from typing import Any

import cv2
import torch
import numpy as np
from glob import glob

from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.core.config import cfg as tracker_cfg

from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg


class DroneDetectTrack():
    def __init__(self, video_path) -> None:
        self.detect_cfg = get_cfg()
        self.video_path = video_path
        self.detect_cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        self.detect_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.detect_cfg.INPUT.MAX_SIZE_TEST = 1024
        self.detect_cfg.MODEL.DEVICE = "cuda:0"
        self.detect_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
        self.detect_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01
        self.detect_cfg.MODEL.WEIGHTS = "../weights/model_maskrcnn.pth"
        self.detect_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.detector = DefaultPredictor(self.detect_cfg)

        tracker_cfg.merge_from_file("siamrpn_r50_l234_dwxcorr.yaml")
        self.tracker = ModelBuilder()
        self.tracker.load_state_dict(torch.load("../weights/siamrpn_r50_l234_dwxcorr.pth"))
        self.tracker.eval().to(self.detect_cfg.MODEL.DEVICE)
        self.tracker = build_tracker(self.tracker)

    def DetectDrone(self, frame):
        boxes = []
        outputs = self.detector(frame)
        boxes_t = outputs["instances"].pred_boxes.to("cpu")
        boxes = [box_t.numpy() for box_t in boxes_t]
        return boxes
    
    def DroneTrack(self, frame):
        outputs = self.tracker.track(frame)
        return outputs
    
    def __call__(self):
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        cap = cv2.VideoCapture(self.video_path)
        video_name = self.video_path.split("/")[-1].split(".")[0]
        save_dir = os.path.join("output", video_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, video_name+".avi")
        out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640,512))

        
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        
        # Read until video is completed
        frame_id = 0
        start_tracking = False
        print(f"\n\n****START PROCESSING VIDEO: {video_name}...***\n\n")
        while(cap.isOpened()):
            frame_id += 1
            print(f"FRAME: {frame_id}")
            ret, frame = cap.read()
            if ret:
                if not start_tracking:
                    boxes = self.DetectDrone(frame)
                if len(boxes):
                    if not start_tracking:
                        box = boxes[0]
                        box = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
                        self.tracker.init(frame, box)
                    boxes = self.DroneTrack(frame)
                    start_tracking = True
                if len(boxes):
                    box = boxes['bbox']
                    box = [int(coord) for coord in box]
                    frame = cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 255), 1)
                out.write(frame)
                cv2.waitKey(1)
            else:
                break
        # When everything done, release the video capture object
        cap.release()
        out.release()
        
        # Closes all the frames
        cv2.destroyAllWindows()
        print(f"\nVIDEO SAVED IN: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default="../dataset/video/Test01.mp4")
    opt = parser.parse_args()
    drone_detect = DroneDetectTrack(opt.video)
    drone_detect()
        