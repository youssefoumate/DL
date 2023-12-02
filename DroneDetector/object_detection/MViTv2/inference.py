import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

class DroneInference():
    """This class detects defects in the image
    """
    def __init__(self, video_path):
        """initalize model configurations
        Args:
            opt (ArgumentParser): command line arguments
        """
        self.video_path = video_path
        self.cfg = LazyConfig.load(opt.config_file)
        self.predictor = instantiate(self.cfg.model)
        self.predictor.to(self.cfg.train.device)
        self.predictor.eval()
        checkpointer = DetectionCheckpointer(self.predictor)
        checkpointer.load("../../weights/model_mvitv2.pth")
        self.root_input_folder = "input"
        #self.aug = T.ResizeShortestEdge(
        #    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        #)
    
    def predict(self, original_image):
        with torch.no_grad():
            image = original_image[:, :, ::-1]
            height, width = image.shape[:2]
            #image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to(self.cfg.train.device)

            inputs = {"image": image, "height": height, "width": width}

            predictions = self.predictor([inputs])[0]
            return predictions

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
                    outputs = self.predict(frame)
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
    parser.add_argument('--video', type=str, default="../../dataset/video/Test02.mp4")
    parser.add_argument('--config_file', type=str, default="config.py")
    opt = parser.parse_args()
    drone_detect = DroneInference(opt.video)
    drone_detect()