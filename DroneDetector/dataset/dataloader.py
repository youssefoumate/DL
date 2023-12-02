import os
import numpy as np
import cv2
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import random
from tqdm import tqdm
random.seed(10)

class DroneDataset():
    def __init__(self, label_dir, img_dir) -> None:
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.images = []
        self.labels = []
    def load_labels(self):
        dataset_labels = []
        label_files = os.listdir(self.label_dir)
        for label in label_files:
            label_path = os.path.join(self.label_dir, label)
            with open(label_path, "r") as f:
                coords = f.readline()
                if coords == "":
                    continue
                coords = coords.replace("\n", "").split(" ")
                coords = [float(coord) for coord in coords]
                self.labels.append(coords)
            img_path = os.path.join(self.img_dir, label.split(".")[0]+".JPEG")
            self.images.append(img_path)
        return dataset_labels

    def drone_dicts(self, mode="onlytrain"):
        """Dataloading helping function
        
        Args:
            img_dir (str): path to data
            mode (str): training or validation modes
        
        Returns:
            list: list of annotations and input dicts 
        """
        self.load_labels()
        dataset_dicts = []
        for idx, (label, img_path) in tqdm(enumerate(zip(self.labels, self.images))):
            record = {}
            record["file_name"] = img_path
            image = cv2.imread(img_path)
            if image is None:
                print("failed to read img")
                continue
            record["height"], record["width"] = image.shape[:2]
            record["image_id"] = int(idx)
            bbox = [(label[1] - label[3]/2)*record["width"], 
                         (label[2] - label[4]/2)*record["height"], 
                         label[3]*record["width"],
                         label[4]*record["height"]]
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": [[bbox[0], 
                                 bbox[1], 
                                 bbox[0], 
                                 bbox[1]+bbox[3],
                                 bbox[0]+bbox[2],
                                 bbox[1]+bbox[3], 
                                 bbox[0]+bbox[2],
                                 bbox[1]]],
                "category_id": 0
            }
            record["annotations"] = [obj]
            if record["annotations"]:
                if mode == "train":
                    if idx <= len(self.images) * 0.95:
                        dataset_dicts.append(record)
                elif mode == "val":
                    if idx > len(self.images) * 0.95:
                        dataset_dicts.append(record)
        random.shuffle(dataset_dicts)
        return dataset_dicts

    def drone_dataset_metadata(self):
        """register dataset
        
        Args:
            img_dir (str): path to data
        
        Returns:
            MetadataCatalog: dataset metadata
        """
        for mode in ["train", "val"]:
            DatasetCatalog.register("drone_dataset_"+ mode, lambda mode=mode: self.drone_dicts(mode))
            MetadataCatalog.get("drone_dataset_"+ mode).set(thing_classes=["drone"])
            MetadataCatalog.get("drone_dataset_"+ mode).evaluator_types = "coco"
        return MetadataCatalog.get("drone_dataset_"+mode)

if __name__ == "__main__":
    mode = "train"
    label_dir = "/home/youssef/DroneDetector/dataset/labels"
    img_dir = "/home/youssef/DroneDetector/dataset/images"
    indoor_defect_dataset = DroneDataset(label_dir, img_dir)
    metadata = indoor_defect_dataset.drone_dataset_metadata()
    dataset_dicts = indoor_defect_dataset.drone_dicts(mode)
    labels = []
    for record in dataset_dicts:
        for anno in record['annotations']:
            labels.append(anno['category_id'])
    classes, counts = np.unique(np.array(labels), return_counts=True)
    print(dict(zip(classes.astype(str), counts)))
    print(len(dataset_dicts))
    for idx, d in enumerate(tqdm(dataset_dicts)):
        print(d["file_name"].split("/")[-1])
        img = cv2.imread(d["file_name"])
        annos = d["annotations"]
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("image", out.get_image()[:, :, ::-1])
        cv2.waitKey()