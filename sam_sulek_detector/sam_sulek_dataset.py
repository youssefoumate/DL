import os
import numpy as np
import cv2
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

class SamSulekDataset():
    """This class detects defects in the image
    """
    def __init__(self):
        """initalize model configurations
        Args:
            opt (ArgumentParser): command line arguments
        """
        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.cfg.INPUT.MAX_SIZE_TEST = 1024
        self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)
        self.root_input_folder = "input"

    def __call__(self, image=None):
        """ This function performs inference on an image using different weights 
        and thresholds for each class.

        Args:
            image (numpy array): of shape (H, W, C), where C is 3.

        Returns:
            classes (list): a list of integers representing the predicted class indices for each instance in the image.
            masks (list): a list of boolean numpy arrays of shape (H, W) representing the predicted masks for each instance in the image.
        """
        for i in range(100, 480500, 100):
            image_name  = f"{i}.jpg"
            if i < 400000:
                split = "TRAINING"
            elif i < 430000:
                split = "VALIDATION"
            else:
                split = "TEST"
            image = cv2.imread(os.path.join(self.root_input_folder, image_name))
            h, w = image.shape[:2]
            outputs = self.predictor(image)
            cls = outputs["instances"].pred_classes.to("cpu")
            boxes_cls = outputs["instances"].pred_boxes.to("cpu")
            person_cls_idx = np.where(cls.numpy() == 0)
            scores = outputs["instances"].scores.to("cpu").numpy()
            for box, score in zip(boxes_cls[person_cls_idx], scores):
                if score > 0.97:
                    box = box.numpy().astype(int)
                    image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 3)
                    with open("sam_sulek_annos.csv", "a") as f:
                        f.writelines(f"{split},{image_name},sam,{box[0]/w},{box[1]/h},,,{box[2]/w},{box[3]/h},,\n")
            #cv2.imshow("image", cv2.resize(image, (720, 480)))
            #cv2.waitKey(1)
            """
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=0.5)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow("sam_sulek_pred", out.get_image()[:, :, ::-1])
            cv2.waitKey(2)
            """


if __name__ == "__main__":
    sam_sulek_dataset = SamSulekDataset()
    sam_sulek_dataset()