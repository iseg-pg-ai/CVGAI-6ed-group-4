import cv2
from ultralytics import YOLO

# Load the YOLOv8n-cls model
model = YOLO("yolov8n-cls.pt")

# Read an image using OpenCV
image = cv2.imread("dog.jpeg")


if image is None:
	print("Error: Image not found or unable to load.")
else:
	# Run classification on the image
	results = model(image)

	# Access the top-1 prediction
	top1_index = results[0].probs.top1  # Get the index of the top-1 predicted class
	top1_conf = results[0].probs.top1conf.item()  # Get the confidence score of the top-1 prediction
	class_name = model.names[top1_index]  # Get the class name using the index

	# Display the result
	print(f"Class: {class_name}, Confidence: {top1_conf:.2f}")
