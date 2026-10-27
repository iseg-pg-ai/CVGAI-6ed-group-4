import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

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
		h, w, c = frame.shape
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = face_mesh.process(rgb)

		if results.multi_face_landmarks:
			for face_landmarks in results.multi_face_landmarks:
				# Example key landmarks (from the 468 indices)
				left_eye = face_landmarks.landmark[33]   # left eye outer corner
				right_eye = face_landmarks.landmark[263] # right eye outer corner
				nose_tip = face_landmarks.landmark[1]
				mouth_center = face_landmarks.landmark[13]

				# Convert to pixel coords
				coords = [
					("Olho Esq", int(left_eye.x * w), int(left_eye.y * h)),
					("Olho Dir", int(right_eye.x * w), int(right_eye.y * h)),
					("Nariz", int(nose_tip.x * w), int(nose_tip.y * h)),
					("Boca", int(mouth_center.x * w), int(mouth_center.y * h))
				]

				for label, x, y in coords:
					cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
					cv2.putText(frame, label, (x, y - 10),
								cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

				#  draw mesh connections
				mp_draw.draw_landmarks(
					frame, face_landmarks,
					mp_face.FACEMESH_CONTOURS
				)

		cv2.imshow("Face Keypoints", frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

cap.release()
cv2.destroyAllWindows()
