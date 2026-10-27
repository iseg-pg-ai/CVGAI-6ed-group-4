import cv2
from ultralytics import YOLO

# Load pretrained YOLOv8n 
model = YOLO("yolov8n.pt")

# Read the image 
source = cv2.imread("pessoascarros.jpg")
if source is None:
	print("Error: Image not found or unable to load.")
else:
	# Run inference on the source
	results = model(source)

	# Accessing detections from the results
	for result in results:
		boxes = result.boxes
		for box in boxes:
			x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
			conf = box.conf[0].item()
			cls = int(box.cls[0].item())
			class_name = model.names[cls]

			# Draw bounding box and label
			if(class_name == "car"):
			    cv2.rectangle(source, (x1, y1), (x2, y2), (0, 255, 0), 2)
			    label = f"{class_name}: {conf:.2f}"
			    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
			    cv2.rectangle(source, (x1, y1 - text_height - 4), (x1 + text_width, y1), (0, 255, 0), -1)
			    cv2.putText(source, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
			else:
				cv2.rectangle(source, (x1, y1), (x2, y2), (0, 255, 0), 2)
				label = f"{class_name}: {conf:.2f}"
				(text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
				cv2.rectangle(source, (x1, y1 - text_height - 4), (x1 + text_width, y1), (0, 0, 255), -1)
				cv2.putText(source, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

	cv2.imshow("YOLOv8", source)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
