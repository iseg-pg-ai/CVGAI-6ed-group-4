import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
tipIds = [4, 8, 12, 16, 20]

while True:
	success, img = cap.read()
	img = cv2.flip(img, 1)
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = hands.process(imgRGB)

	if results.multi_hand_landmarks:
		for handLms in results.multi_hand_landmarks:
			lmList = []
			h, w, c = img.shape
			for id, lm in enumerate(handLms.landmark):
				cx, cy = int(lm.x * w), int(lm.y * h)
				lmList.append((id, cx, cy))

			fingers = []
			if lmList:
				# Thumb
				fingers.append(1 if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1] else 0)
				# Other fingers
				for id in range(1, 5):
					fingers.append(1 if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2] else 0)

				totalFingers = fingers.count(1)

				if totalFingers == 0:
					state = "OFF"
				elif totalFingers == 5:
					state = "ON"
				else:
					state = "..."

				cv2.putText(img, f'State: {state}', (10, 70),
							cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

			mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

	cv2.imshow("Gesture Control", img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
