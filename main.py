import numpy as np
from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import *

results = {}

mot_tracker = Sort()

# load model detect car
coco_model = YOLO('yolov8n.pt')

# load model license plate detector
license_plate_detector = YOLO('./license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./sample_5.mp4')

# from coco.names: https://github.com/pjreddie/darknet/blob/master/data/coco.names
# car, motorbike, bus, truck
vehicles = [2, 3, 5, 7]

# read frames
frame_numb = 1
ret = True
while ret:
    frame_numb += 1
    ret, frame = cap.read()
    if ret and frame_numb < 100:
        results[frame_numb] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_bounding_boxes = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_bounding_boxes.append([x1, y1, x2, y2, score])

        # # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_bounding_boxes))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            # crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 160, 255, cv2.THRESH_BINARY)

            try:
                cv2.imshow('threshold', license_plate_crop_thresh)
                cv2.waitKey(2000)
            except Exception as e:
                print("Error: {}".format(e))
            cv2.destroyAllWindows()

            # read license plate number
            license_plate_text, license_plate_text_score = util.read_license_plate(license_plate_crop_thresh)

            if license_plate_text is not None:
                results[frame_numb][car_id] = {
                    'car': {
                        'bbox': [xcar1, ycar1, xcar2, ycar2]
                    },
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    }
                }
    else:
        break
# write results:
write_csv(results, './test.csv')

