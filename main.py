import json

import numpy as np
from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import *

results = {}
car_results = {}

mot_tracker = Sort()

# load model detect car
coco_model = YOLO('model/yolov8n.pt')

# load model license plate detector
license_plate_detector = YOLO('model/license_plate_detector.pt')

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
    results[frame_numb] = {}
    if ret:
        print(frame_numb)
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_bounding_boxes = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles and x1 and y1 and x2 and y2:
                detections_bounding_boxes.append([x1, y1, x2, y2, score])

        # # track vehicles
        if not detections_bounding_boxes:
            detections_bounding_boxes.append([0, 0, 0, 0, 0])
        track_ids = mot_tracker.update(np.asarray(detections_bounding_boxes))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            # crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # # Apply image enhancement (e.g., histogram equalization)
            # license_plate_crop_enhanced = cv2.equalizeHist(cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY))
            #
            # # Apply adaptive thresholding
            # license_plate_crop_thresh = cv2.adaptiveThreshold(license_plate_crop_enhanced, 255,
            #                                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

            # process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 160, 230, cv2.THRESH_BINARY)

            # try:
            #     cv2.imshow('threshold', license_plate_crop_thresh)
            #     cv2.waitKey(2000)
            # except Exception as e:
            #     print("Error: {}".format(e))
            # cv2.destroyAllWindows()

            # read license plate number
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
            license_plates = []
            if license_plate_text is not None and [xcar1, ycar1, xcar2, ycar2] != [-1, -1, -1, -1]:
                results[frame_numb][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                               'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                 'text': license_plate_text,
                                                                 'bbox_score': score,
                                                                 'text_score': license_plate_text_score}}
                if car_id not in car_results.keys():
                    car_results[car_id] = {
                        'car': {
                            'bbox': [xcar1, ycar1, xcar2, ycar2],
                            'license_plates': [],
                        }
                    }
                car_results[car_id]['car']['license_plates'].append({
                    'bbox': [x1, y1, x2, y2],
                    'text': license_plate_text,
                    'bbox_score': score,
                    'text_score_line_1': license_plate_text_score[0],
                    'text_score_line_2': license_plate_text_score[1]
                })
    else:
        break

final_lps = shorten_result(car_results)

# write results:
write_csv(results, 'result/result_sample1.csv')
write_csv_from_json(final_lps, 'result/shorten_result.csv')

with open('result/full_result.json', 'w') as f:
    json.dump(car_results, f, indent=4)

with open('result/shorten_result.json', 'w') as f:
    json.dump(final_lps, f, indent=4)
