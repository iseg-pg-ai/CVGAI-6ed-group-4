import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

# Eye landmark indices (from FaceMesh)
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

with mp_face.FaceMesh(
	max_num_faces=1,
	refine_landmarks=True,
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5
) as face_mesh:

	while True:
		success, frame = cap.read()
		if not success:
			break

		frame = cv2.flip(frame, 1) 
		h, w, _ = frame.shape
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = face_mesh.process(rgb)

		status_text = ""

		if results.multi_face_landmarks:
			for face_landmarks in results.multi_face_landmarks:
				# Key points
				left_eye = face_landmarks.landmark[33]
				right_eye = face_landmarks.landmark[263]
				nose = face_landmarks.landmark[1]
				chin = face_landmarks.landmark[152]

				# Convert to pixels
				left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
				right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)
				nose_x, nose_y = int(nose.x * w), int(nose.y * h)
				chin_y = int(chin.y * h)
				eye_mid_x = (left_eye_x + right_eye_x) // 2
				eye_mid_y = (left_eye_y + right_eye_y) // 2

				# ---------- HORIZONTAL ----------
				if nose_x > eye_mid_x + 20:  
					status_text = "Looking Right"
				elif nose_x < eye_mid_x - 20:
					status_text = "Looking Left"
				else:
					# ---------- VERTICAL ----------
					dist_eye_chin = chin_y - eye_mid_y
					dist_eye_nose = nose_y - eye_mid_y

					ratio = dist_eye_nose / dist_eye_chin

					if ratio > 0.65:
						status_text = "Looking Down"
					elif ratio < 0.35:
						status_text = "Looking Forward"
					else:
						status_text = "Looking Down"

				# ---------- BLINK DETECTION ----------
				left_dist = abs(int(face_landmarks.landmark[LEFT_EYE_TOP].y * h) -
								int(face_landmarks.landmark[LEFT_EYE_BOTTOM].y * h))
				right_dist = abs(int(face_landmarks.landmark[RIGHT_EYE_TOP].y * h) -
								 int(face_landmarks.landmark[RIGHT_EYE_BOTTOM].y * h))

				if left_dist < 6 and right_dist < 6:  
					status_text += " + BLINK"

				# Draw keypoints
				cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 255), -1)
				cv2.circle(frame, (left_eye_x, left_eye_y), 5, (255, 0, 0), -1)
				cv2.circle(frame, (right_eye_x, right_eye_y), 5, (255, 0, 0), -1)

		# Show status
		cv2.putText(frame, status_text, (30, 50),
					cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

		cv2.imshow("Face Orientation + Blink Detection", frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

cap.release()
cv2.destroyAllWindows()
