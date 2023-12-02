# DroneDetector
This repo test multiple methods to detect flying drones

## Demo

https://github.com/youssefoumate/DroneDetector/assets/17303586/c6061a36-41f5-4521-9682-62a83492682a


## Usage
### Dependencies
- Python==3.10.11
- Pytorch==2.0.1
- Detectron2 
```
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'@v0.6
```
- OpenCV
- ultralytics

### FILES
Please Download the following

- [weights](https://drive.google.com/drive/folders/1gZQebZMim6188AhvPmqpeYJzglrUK7Ss?usp=sharing) (DroneDetection/weights)
- [dataset](https://drive.google.com/file/d/1ewy8Ij_hW56pWMXUPCCmYW1FB4QoIVw9/view?usp=sharing) (DroneDetection/dataset)

## OBJECT DETECTION
### YOLO (BEST FOR LONG TERM DETECTION)
- Train

```
user@server:~/DroneDetector$ cd object_detection/YOLO
user@server:~/DroneDetector/object_detection/YOLO$ python3 train.py
```
- Inference

```
user@server:~/DroneDetector/object_detection/YOLO$ yolo task=detect mode=predict model=../../weights/model_yolov8l.pt source=../../dataset/video/Test02.mp4 show=True imgsz=640 name=yolov8n_infer hide_labels=True conf=0.005
```
### MASK RCNN
- Train

```
user@server:~/DroneDetector cd object_detection/maskRCNN
user@server:~/DroneDetector/object_detection/maskRCNN$ python3 train.py
```
- Inference

```
user@server:~/DroneDetector/object_detection/maskRCNN$ python3 inference.py --video [video_path]
```
## OBJECT DETECTION AND TRACKING (BEST FOR SHORT TERM DETECTION/TRACKING)
- Inference

```

user@server:~/DroneDetector$ cd object_detection/object_detect_track
user@server:~/DroneDetector/object_detection$ export PYTHONPATH=./pysot:$PYTHONPATH
user@server:~/DroneDetector/object_detection$ python3 detect_track.py --video [video_path]
```
