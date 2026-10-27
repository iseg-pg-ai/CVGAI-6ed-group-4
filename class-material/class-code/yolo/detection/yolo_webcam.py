import cv2
from ultralytics import YOLO

# Load pretrained YOLOv8n
model = YOLO("yolov8n.pt")

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

	# Run inference on the current frame
	results = model(frame)

	# Draw detections
	for result in results:
		boxes = result.boxes
		for box in boxes:
			x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
			conf = box.conf[0].item()
			cls = int(box.cls[0].item())
			class_name = model.names[cls]

			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
			label = f"{class_name}: {conf:.2f}"
			(tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 1)
			cv2.rectangle(frame, (x1, max(0, y1 - th - 4)), (x1 + tw, y1), (0, 0, 255), -1)
			cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 1)

	# Show the result
	cv2.imshow("YOLOv8 Webcam", frame)

	# Press 'q' to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
