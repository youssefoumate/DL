from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8l.pt')
 
# Training.
results = model.train(
   data='drone_detect.yaml',
   imgsz=1024,
   epochs=100,
   batch=8,
   name='yolov8n_custom_large')