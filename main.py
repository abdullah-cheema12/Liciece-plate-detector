from ultralytics import YOLO #detecting objects
import cv2 #opemcv impage processing
import numpy as np #array managment
from sort.sort import Sort #ja us ne frame to frame check karna hai tu ye ylo ki madad se detect kare ga 
import util #get all data from cars
from util import get_car, read_license_plate, write_csv
import matplotlib.pyplot as plt #video opener

results = {} #array to store data
  
mot_tracker = Sort() #object traking variable

# Load models
coco_model = YOLO('yolov8n.pt') #pretrained model for tracking objects cumming from yolo lib
license_plate_detector = YOLO('./license_plate_detector.pt') # pretrained model for tracking num plates

# Load video
cap = cv2.VideoCapture('./test-sample.mp4') #video loader 

vehicles = [2, 3, 5, 7]  # Example vehicle classes: car, motorcycle, bus, truck ids for objects 

# Create a figure for Matplotlib #making a outline on cars and plates
plt.ion() 
fig, ax = plt.subplots()

# Read frames frames detctor
frame_nmr = -1 
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Draw vehicle bounding boxes and IDs
        for track in track_ids:
            tx1, ty1, tx2, ty2, track_id = track
            cv2.rectangle(frame, (int(tx1), int(ty1)), (int(tx2), int(ty2)), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {int(track_id)}', (int(tx1), int(ty1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
                    
                    # Draw license plate bounding box and text with larger font size
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, license_plate_text, (int(x1), int(y1)-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Display the frame using Matplotlib
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.draw()
        plt.pause(0.001)
        ax.clear()

        # Break the loop if 'space' is pressed
        if cv2.waitKey(1) & 0xFF == 32:
            break

# Release the video capture and close all OpenCV windows 
cap.release()
plt.close()

# Write results to CSV database for stoing number plates data
write_csv(results, './test.csv')
