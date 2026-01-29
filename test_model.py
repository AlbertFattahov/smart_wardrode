
from ultralytics import YOLO
model = YOLO('models/exp19/weights/best.pt')
#model.predict(source="output_yolo/images/val", save=True, conf=0.25)

model.predict("test_images/", save=True, conf=0.5)
