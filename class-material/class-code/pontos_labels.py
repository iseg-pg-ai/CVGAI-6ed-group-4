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
			h, w, c = frame.shape
			landmarks = results.pose_landmarks.landmark

			# Left and Right landmarks
			left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
			right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
			left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
			right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
			left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
			right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

			# Draw text labels
			cv2.putText(frame, "Ombro", (int(left_shoulder.x * w), int(left_shoulder.y * h) - 10),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
			cv2.putText(frame, "Ombro", (int(right_shoulder.x * w), int(right_shoulder.y * h) - 10),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

			cv2.putText(frame, "Cotovelo", (int(left_elbow.x * w), int(left_elbow.y * h) - 10),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
			cv2.putText(frame, "Cotovelo", (int(right_elbow.x * w), int(right_elbow.y * h) - 10),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

			cv2.putText(frame, "Pulso", (int(left_wrist.x * w), int(left_wrist.y * h) - 10),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
			cv2.putText(frame, "Pulso", (int(right_wrist.x * w), int(right_wrist.y * h) - 10),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

			# Draw skeleton
			mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

		cv2.imshow("Pose Estimation", frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

cap.release()
cv2.destroyAllWindows()
