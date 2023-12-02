import os
import sys
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
sys.path.append("../..")
from dataset.dataloader import DroneDataset
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

class Trainer(DefaultTrainer):
    """ Class that trains the model
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """return an evaluator for validation step

        Args:
            cfg(detectron2.config.CfgNode): evaluation configs
            dataset_name(str): registered dataset name
            output_folder(str): results output folder
        Returns:
            COCOEvaluator: class used to evaluate model using coco metrics
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
    
class DroneDetectTrainer():
    def __init__(self, label_dir, img_dir) -> None:
        self.label_dir_list = label_dir
        self.img_dir_dict = img_dir
        self.drone_dataset = DroneDataset(label_dir, img_dir)
        self.metadata = self.drone_dataset.drone_dataset_metadata()

    def set_up_train(self, mode="val"):
        """set up training configurations

        Args:
            img_dir(str): path to image data
        Returns:
            detectron2.config.CfgNode: training configurations
        """
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("drone_dataset_train",)
        if mode == "val":
            cfg.DATASETS.TEST = ("drone_dataset_val",)
            cfg.TEST.EVAL_PERIOD = 1000
        else:
            cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.SOLVER.CHECKPOINT_PERIOD = 1000
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.0001
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.SOLVER.MAX_ITER = 100000
        cfg.SOLVER.STEPS = []
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.OUTPUT_DIR = "output_mrcnnx101_drone"
        cfg.MODEL.DEVICE = "cuda:0"
        return cfg

    def set_up_val(self):
        """set up evaluation configurations

        Returns:
            detectron2.config.CfgNode: evaluation configurations
        """
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.DEVICE = "cuda:0"
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        return cfg
        
    def train(self):
        """training function
        """
        cfg = self.set_up_train(mode='train')
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=True)
        trainer.train()

    def eval(self):
        """evaluation function
        """
        cfg = self.set_up_val()
        evaluator = COCOEvaluator("drone_dataset_val", output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
        val_loader = build_detection_test_loader(cfg, "drone_dataset_val")
        cfg.DATASETS.TRAIN = ("drone_dataset_train",)
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=True)
        predictor = DefaultPredictor(cfg)
        print(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == "__main__":
    label_dir = "/home/youssef/DroneDetector/dataset/labels"
    img_dir = "/home/youssef/DroneDetector/dataset/images"
    indoor_defect_trainer = DroneDetectTrainer(label_dir, img_dir)
    indoor_defect_trainer.train()
