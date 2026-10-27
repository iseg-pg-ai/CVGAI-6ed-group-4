import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

canvas = None
prev_x, prev_y = 0, 0

while True:
	success, img = cap.read()
	img = cv2.flip(img, 1)
	if canvas is None:
		canvas = np.zeros_like(img)

	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = hands.process(imgRGB)

	if results.multi_hand_landmarks:
		for handLms in results.multi_hand_landmarks:
			h, w, c = img.shape
			for id, lm in enumerate(handLms.landmark):
				cx, cy = int(lm.x * w), int(lm.y * h)

				# Index fingertip
				if id == 8:
					cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
					if prev_x == 0 and prev_y == 0:
						prev_x, prev_y = cx, cy
					cv2.line(canvas, (prev_x, prev_y), (cx, cy), (255, 0, 0), 5)
					prev_x, prev_y = cx, cy

			mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

	# Combine drawing with camera feed
	img = cv2.addWeighted(img, 1, canvas, 0.5, 0)

	cv2.imshow("Air Painter", img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
