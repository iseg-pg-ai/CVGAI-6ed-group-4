from ultralytics import YOLO
model = YOLO("yolov8n-cls.pt")

for class_id, class_name in model.names.items():
	print(f"{class_id}: {class_name}")