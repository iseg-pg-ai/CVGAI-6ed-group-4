import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,
				  min_tracking_confidence=0.5) as pose:

	while True:
		success, frame = cap.read()
		if not success:
			break

		frame = cv2.flip(frame, 1)  # mirror view
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = pose.process(rgb)

		if results.pose_landmarks:
			mp_drawing.draw_landmarks(
				frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
			)

		cv2.imshow("Pose Estimation", frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

cap.release()
cv2.destroyAllWindows()
