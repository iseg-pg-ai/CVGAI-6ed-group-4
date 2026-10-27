import cv2
from ultralytics import YOLO

# Load the YOLOv8n-pose model
model = YOLO("yolov8n-pose.pt")

# Read the image 
image = cv2.imread("yoga.jpg")

if image is None:
	print("Error: Image not found or unable to load.")
else:
	# Run pose estimation on the image
	results = model(image)

	# Iterate through each detected person in the results
	for i, result in enumerate(results):
		# Access the xy attribute to get the 17 (x, y) coordinates
		keypoints = result.keypoints.xy[0] 
	
		# Iterate through each keypoint and print the x, y coordinates
		for j, (x, y) in enumerate(keypoints):
			print(f"Keypoint {j+1}: (x={x:.2f}, y={y:.2f})")

	#draw the keypoints and skeleton on the image
	annotated_image = results[0].plot()

	# Display the result with keypoints
	cv2.imshow("YOLOv8 Pose", annotated_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
