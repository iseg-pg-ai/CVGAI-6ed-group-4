import cv2
import mediapipe as mp

# Initialize mediapipe drawing + face mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Configure FaceMesh
with mp_face_mesh.FaceMesh(max_num_faces=2,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:

	while cap.isOpened():
		success, frame = cap.read()
		if not success:
			print("Ignoring empty frame...")
			continue

		# Convert frame from BGR (OpenCV) to RGB (MediaPipe)
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# Process the frame
		results = face_mesh.process(rgb_frame)

		# Convert back to BGR for OpenCV display
		frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

		# Draw landmarks
		if results.multi_face_landmarks:
			for face_landmarks in results.multi_face_landmarks:
				mp_drawing.draw_landmarks(
					image=frame,
					landmark_list=face_landmarks,
					connections=mp_face_mesh.FACEMESH_TESSELATION,
					landmark_drawing_spec=None,
					connection_drawing_spec=mp_drawing.DrawingSpec(
						color=(0,255,0), thickness=1, circle_radius=1
					)
				)

		# Show the frame
		cv2.imshow("MediaPipe FaceMesh", frame)

		# Press 'q' to exit
		if cv2.waitKey(5) & 0xFF == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()
