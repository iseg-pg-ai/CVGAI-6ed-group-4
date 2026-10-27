from PIL import Image, ImageDraw
import face_recognition
import cv2
import numpy as np
import os
from pathlib import Path

video_capture = cv2.VideoCapture(0)

#-----------------------------------------------------------------------------
"""
cr_image = face_recognition.load_image_file("Cristiano_Ronaldo.jpg")
cr_face_encoding = face_recognition.face_encodings(cr_image)[0]

gr_image = face_recognition.load_image_file("Georgina_Rodrigues.jpg")
gr_face_encoding = face_recognition.face_encodings(gr_image)[0]

ed_image = face_recognition.load_image_file("Antonio_Costa.jpg")
ed_face_encoding = face_recognition.face_encodings(ed_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [cr_face_encoding,gr_face_encoding,ed_face_encoding]
known_face_names = ["Cristiano Ronaldo","Georgina Rodriguez","Antonio Costa"]	
"""
#-----------------------------------------------------------------------------
known_face_encodings = []
known_face_names = []

extensions = {".jpg", ".jpeg", ".png"}

for image_path in Path(".").iterdir():
	if image_path.suffix.lower() not in extensions:
		continue

	image = face_recognition.load_image_file(str(image_path))
	face_encodings = face_recognition.face_encodings(image)

	if not face_encodings:
		print(f"No face found in {image_path.name}")
		continue

	# Get name from filename
	name = image_path.stem.replace("_", " ")

	# Add encoding and name to the lists
	known_face_encodings.append(face_encodings[0])
	known_face_names.append(name)

	print(f"Loaded: {name}")

print(f"\nTotal known faces loaded: {len(known_face_names)}")

#-----------------------------------------------------------------------------


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
	# Grab a single frame of video
	ret, frame = video_capture.read()

	# Only process every other frame of video to save time
	if process_this_frame:
		# Resize frame of video to 1/4 size for faster face recognition processing
		small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

		
		rgb_small_frame = small_frame
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
		face_names = []
		
		for face_encoding in face_encodings:
			# See if the face is a match for the known face(s)
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
			name = "Unknown"
			face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:
				name = known_face_names[best_match_index]
			face_names.append(name)
			print("A procurar")
	process_this_frame = not process_this_frame


	# Display the results
	
	for (top, right, bottom, left), name in zip(face_locations, face_names):
		# Scale back up face locations since the frame we detected in was scaled to 1/4 size
		top *= 4
		right *= 4
		bottom *= 4
		left *= 4

		# Draw a box around the face
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

		# Draw a label with a name below the face
		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		name =  name.replace("_", " ")
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
		
		
		
	
	
	# Display the resulting image
	cv2.imshow('Video', frame)

	# Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()