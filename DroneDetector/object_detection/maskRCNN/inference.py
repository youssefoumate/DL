import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
from config import infer_setup

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
        self.cfg = infer_setup(self.cfg, opt)
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
    parser.add_argument('--device', type=str, default="cuda:0")
    opt = parser.parse_args()
    drone_detect = DroneInference(opt.video)
    drone_detect()