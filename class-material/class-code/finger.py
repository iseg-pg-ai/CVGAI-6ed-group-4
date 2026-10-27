import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Landmark indices for fingertips
tip_ids = [4, 8, 12, 16, 20]

while True:
	success, img = cap.read()
	img = cv2.flip(img, 1)
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = hands.process(img_rgb)

	if results.multi_hand_landmarks:
		for hand_landmarks, hand_handedness in zip(
			results.multi_hand_landmarks, results.multi_handedness
		):
			lm_list = []
			h, w, c = img.shape
			for id, lm in enumerate(hand_landmarks.landmark):
				cx, cy = int(lm.x * w), int(lm.y * h)
				lm_list.append((id, cx, cy))

			fingers = []
			handedness = hand_handedness.classification[0].label

			if lm_list:
				
				if handedness == "Right":
					if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
						fingers.append(1)
					else:
						fingers.append(0)
				else:  # Left hand
					if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]:
						fingers.append(1)
					else:
						fingers.append(0)

				
				for id in range(1, 5):
					if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
						fingers.append(1)
					else:
						fingers.append(0)

				total_fingers = sum(fingers)

				# label each fingertip with 0/1
				for idx, up in zip(tip_ids, fingers):
					cx, cy = lm_list[idx][1], lm_list[idx][2]
					cv2.putText(img, str(up), (cx, cy-20),
								cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

				cv2.putText(img, f'Fingers: {total_fingers}', (10, 70),
							cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

			mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

	cv2.imshow("Finger Counter", img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
