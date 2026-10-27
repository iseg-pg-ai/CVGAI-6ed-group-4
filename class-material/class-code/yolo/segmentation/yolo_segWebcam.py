import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")

# Open webcam 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
	print("Error: Could not open webcam.")
	raise SystemExit

while True:
	ret, frame = cap.read()
	if not ret:
		print("Error: Failed to grab frame from webcam.")
		break

	# Run inference on the frame
	results = model(frame)

	for result in results:
		masks = result.masks  # Segmentation masks
		boxes = result.boxes  # Bounding boxes
		classes = result.boxes.cls  # Detected classes
		scores = result.boxes.conf  # Confidence scores

		# Draw masks and bounding boxes
		for i in range(len(boxes)):
			# Extract bounding box
			x1, y1, x2, y2 = map(int, boxes[i].xyxy[0].tolist())
			conf = scores[i].item()
			cls = int(classes[i].item())
			class_name = model.names[cls]

			# Draw bounding box
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

			# Label with class and confidence
			label = f"{class_name}: {conf:.2f}"
			(tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
			cv2.rectangle(frame, (x1, max(0, y1 - th - 4)), (x1 + tw, y1), (0, 255, 0), -1)
			cv2.putText(frame, label, (x1, y1 - 4),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

			# Extract mask for current object
			mask = masks.data[i].cpu().numpy()
			mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
			mask_resized = (mask_resized > 0.5).astype(np.uint8)

			# Create colored mask overlay
			colored_mask = np.zeros_like(frame)
			colored_mask[mask_resized > 0] = [0, 0, 255]  # Red mask

			# Blend mask with frame
			frame = cv2.addWeighted(frame, 1, colored_mask, 1, 0)

	# Show result
	cv2.imshow("YOLOv8 Segmentation Webcam", frame)

	# Press 'q' to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
