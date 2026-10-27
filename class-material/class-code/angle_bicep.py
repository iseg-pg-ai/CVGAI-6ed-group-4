import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
	"""Returns the angle (in degrees) at point b given three points."""
	a = np.array(a)  # Shoulder
	b = np.array(b)  # Elbow
	c = np.array(c)  # Wrist

	radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
			  np.arctan2(a[1]-b[1], a[0]-b[0])
	angle = np.abs(radians*180.0/np.pi)
	if angle > 180:
		angle = 360 - angle
	return angle

cap = cv2.VideoCapture(0)

counter = 0
stage = None  # "up" or "down"

with mp_pose.Pose(min_detection_confidence=0.5,
				  min_tracking_confidence=0.5) as pose:

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		frame = cv2.flip(frame, 1)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = pose.process(rgb)

		if results.pose_landmarks:
			landmarks = results.pose_landmarks.landmark
			h, w, c = frame.shape

			# Get keypoints
			shoulder = [int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
						int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)]
			elbow = [int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w),
					 int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h)]
			wrist = [int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w),
					 int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h)]

			# Calculate angle
			angle = calculate_angle(shoulder, elbow, wrist)

			# Draw landmarks + angle text
			cv2.putText(frame, str(int(angle)), tuple(elbow),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

			# Curl counter logic
			if angle > 160:
				stage = "down"
			if angle < 40 and stage == "down":
				stage = "up"
				counter += 1

			cv2.putText(frame, f"Curls: {counter}", (10, 50),
						cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3)

			mp_drawing.draw_landmarks(
				frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
			)

		cv2.imshow("Bicep Curl Counter", frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

cap.release()
cv2.destroyAllWindows()
