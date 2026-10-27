import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
	print("Error: Could not open webcam.")
	raise SystemExit

while True:
	ret, frame = cap.read()
	if not ret:
		print("Error: Failed to grab frame from webcam.")
		break

	# Run pose estimation on the frame
	results = model(frame, verbose=False)
	res = results[0]

	# Draw keypoints/skeleton
	annotated_frame = res.plot()

	# access keypoints for the first detected person
	if res.keypoints is not None and res.keypoints.xy is not None:
		kxy = res.keypoints.xy  # shape: (num_people, num_kpts, 2)
		if kxy.shape[0] > 0:    # at least one person
			first_person_kpts = kxy[0].cpu().numpy()  # (num_kpts, 2)
			for j, (x, y) in enumerate(first_person_kpts, start=1):
				print(f"Keypoint {j}: (x={x:.2f}, y={y:.2f})")

	cv2.imshow("YOLOv8 Pose Webcam", annotated_frame)

	# 'q' to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
