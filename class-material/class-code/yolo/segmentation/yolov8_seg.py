import cv2
from ultralytics import YOLO
import numpy as np

#Load model
model = YOLO("yolov8n-seg.pt")

# Read the image
source = cv2.imread("homem.png")

if source is None:
	print("Error: Image not found or unable to load.")
else:
	# Run inference on the source
	results = model(source)  # Get the results

	# detections
	for result in results:
		masks = result.masks  # Segmentation masks
		boxes = result.boxes  # Bounding boxes
		classes = result.boxes.cls  # Detected classes
		scores = result.boxes.conf  # Confidence scores

		# Draw masks and bounding boxes on the image
		for i in range(len(boxes)):
			# Extract the bounding box coordinates (x1, y1, x2, y2)
			x1, y1, x2, y2 = map(int, boxes[i].xyxy[0].tolist())
			conf = scores[i].item()  # Confidence score
			cls = int(classes[i].item())  # Class index
			class_name = model.names[cls]  # Get class name

			# Draw the bounding box
			cv2.rectangle(source, (x1, y1), (x2, y2), (0, 255, 0), 2)

			# Create label with class name and confidence
			label = f"{class_name}: {conf:.2f}"
			(text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
			cv2.rectangle(source, (x1, y1 - text_height - 4), (x1 + text_width, y1), (0, 255, 0), -1)
			cv2.putText(source, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

			# Extract the mask for the current object and resize it to match the image size
			mask = masks.data[i].numpy()  # Convert the mask to numpy array
			mask_resized = cv2.resize(mask, (source.shape[1], source.shape[0]))  # Resize mask to image size

			# Threshold the mask to make sure it's binary
			mask_resized = (mask_resized > 0.5).astype(np.uint8)

			# Create a colored mask to overlay on the image
			colored_mask = np.zeros_like(source)
			colored_mask[mask_resized > 0] = [0, 0, 255]  # Red color for the mask

			# Overlay the mask on the image
			source = cv2.addWeighted(source, 1, colored_mask, 0.9, 0)

	# Display the annotated image with masks and bounding boxes
	cv2.imshow("YOLOv8  Segmentation", source)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
