import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg

class DroneInference():
    """This class detects defects in the image
    """
    def __init__(self, video_path):
        """initalize model configurations
        Args:
            opt (ArgumentParser): command line arguments
        """
        self.video_path = video_path
        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.cfg.INPUT.MAX_SIZE_TEST = 1024
        self.cfg.MODEL.DEVICE = "cuda:0"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01
        self.cfg.MODEL.WEIGHTS = "../../weights/model_maskrcnn.pth"
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.predictor = DefaultPredictor(self.cfg)
        self.root_input_folder = "input"

    def __call__(self):
            cap = cv2.VideoCapture(self.video_path)
            video_name = self.video_path.split("/")[-1].split(".")[0]
            print(f"\n\n****START PROCESSING VIDEO: {video_name}...***\n\n")
            save_dir = os.path.join("output", video_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, video_name+".avi")
            out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640,512))
            if (cap.isOpened()== False): 
                print("Error opening video stream or file")
            frame_id = 0
            while(cap.isOpened()):
                print(f"FRAME: {frame_id}")
                frame_id += 1
                ret, frame = cap.read()
                if ret:
                    outputs = self.predictor(frame)
                    boxes = outputs["instances"].pred_boxes.to("cpu")
                    scores = outputs["instances"].scores.to("cpu").numpy()
                    for box, _ in zip(boxes, scores):
                        box = box.numpy().astype(int)
                        frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 1)
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
    parser.add_argument('--video', type=str, default="../../dataset/video/Test01.mp4")
    opt = parser.parse_args()
    drone_detect = DroneInference(opt.video)
    drone_detect()