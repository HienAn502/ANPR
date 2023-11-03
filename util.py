import string

import cv2
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

reader.detail = 2  # Increase detail for higher accuracy
reader.mag_ratio = 1.6  # Adjust the image magnification ratio
reader.tesseract_layout = 6  # Use PSM 6 (Assume a single uniform block of text)
reader.tesseract_oem = 1  # Use LSTM OCR Engine
reader.lang_list = ['en']

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'Z': '2',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'T': '7',
                    'H': '8',
                    'B': '8'}
dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '2': 'Z',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '7': 'T',
                    '8': 'H'}
int_ = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                        'license_plate' in results[frame_nmr][car_id].keys() and \
                        'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][
                                                                    0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][
                                                                    1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][
                                                                    2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][
                                                                    3]),
                                                            results[frame_nmr][car_id]['license_plate'][
                                                                'bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate'][
                                                                'text_score'])
                            )
        f.close()


def write_csv_from_json(results, output_path):
    with open(output_path, 'w') as file:
        file.write("{}, {}, {}, {}, {}, {}, {}\n".format("vehicle_id", "vehicle_bbox",
                                                         "plate_bbox", "plate_bbox_score",
                                                         "license_number",
                                                         "license_number_score_1", "license_number_score_2"))
        for vehicle in results:
            file.write("{}, {}, {}, {}, {}, {}, {}\n".format(vehicle["vehicle"]["id"],
                                                             "[{} {} {} {}]".format(
                                                                 vehicle["vehicle"]["bbox"][0],
                                                                 vehicle["vehicle"]["bbox"][1],
                                                                 vehicle["vehicle"]["bbox"][2],
                                                                 vehicle["vehicle"]["bbox"][3]),
                                                             "[{} {} {} {}]".format(
                                                                 vehicle["license_plate"]["bbox"][0],
                                                                 vehicle["license_plate"]["bbox"][1],
                                                                 vehicle["license_plate"]["bbox"][2],
                                                                 vehicle["license_plate"]["bbox"][3]),
                                                             vehicle["license_plate"]["bbox_score"],
                                                             vehicle["license_plate"]["text"],
                                                             vehicle["license_plate"]["text_score_line_1"],
                                                             vehicle["license_plate"]["text_score_line_2"]))
        file.close()


def shorten_result(car_results):
    final_lps = []
    for c_id in car_results.keys():
        line_1 = ''
        line_2 = ''
        max_score_1 = 0
        max_score_2 = 0
        final_lp = ''
        bbox = []
        bbox_score = 0
        for car in car_results[c_id]['car']['license_plates']:
            text = car['text'].split(" ")
            if car['text_score_line_1'] > max_score_1:
                max_score_1 = car['text_score_line_1']
                line_1 = text[0]
                bbox = car['bbox']
                bbox_score = car['bbox_score']
            if car['text_score_line_2'] > max_score_2:
                max_score_2 = car['text_score_line_2']
                line_2 = text[1]

        final_lp = f"{line_1} {line_2}"
        final_lps.append({
            'vehicle': {
                'id': c_id,
                'bbox': car_results[c_id]['car']['bbox']
            },
            'license_plate': {
                'bbox': bbox,
                'text': final_lp,
                'bbox_score': bbox_score,
                'text_score_line_1': max_score_1,
                'text_score_line_2': max_score_2,
            }
        })
    return final_lps


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text

    Returns:
        bool: True if the license plate complies with the format, False otherwise
    """
    if len(text) < 8:
        return False, ""

    text = str(text)

    if not (text[0] in int_ or text[0] in dict_char_to_int.keys()):
        return False, ""
    if not (text[1] in int_ or text[1] in dict_char_to_int.keys()):
        return False, ""
    if text[2] not in string.ascii_uppercase and text[2] not in int_:
        text = text[:2] + text[3:]
        if 8 > len(text):
            return False, ""
    if not (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()):
        return False, ""
    for i in [3, 4, 5, 6]:
        if not (text[i] in int_ or text[i] in dict_char_to_int.keys()):
            return False, ""
    if text[7] not in string.ascii_uppercase and text[7] not in int_:
        text = text[:7] + text[8:]
        if 8 > len(text):
            return False, ""
    if not (text[7] in int_ or text[7] in dict_char_to_int.keys()):
        return False, ""
    if len(text) == 9:
        if not (text[8] in int_ or text[8] in dict_char_to_int.keys()):
            return False, ""

    return True, text


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text

    Returns:
        str: Formatted license plate text]
    """
    license_plate = ''
    mapping = {
        0: dict_char_to_int,
        1: dict_char_to_int,
        2: dict_int_to_char,
        3: dict_char_to_int,
        4: dict_char_to_int,
        5: dict_char_to_int,
        6: dict_char_to_int,
        7: dict_char_to_int,
        8: dict_char_to_int
    }
    for i in [0, 1, 2, 3, 4, 5, 6, 7]:
        if text[i] in mapping[i].keys():
            license_plate += mapping[i][text[i]]
        else:
            license_plate += text[i]

    if len(text) == 9:
        if text[8] in mapping[8].keys():
            license_plate += mapping[8][text[8]]
        else:
            license_plate += text[8]

    license_plate = license_plate[:2] + "-" + license_plate[2:4] + " " + license_plate[4:]
    return license_plate


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    detections = reader.readtext(license_plate_crop, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.')
    if len(detections) >= 2:
        text_result = ""
        score_result = []
        for detection in detections:
            bbox, text, score = detection
            score_result.append(score)

            text_result += text.upper().replace(' ', '')

        format_check, check_text = license_complies_format(text_result)
        if format_check:
            text_result = format_license(check_text)
            print("License after formatting:", text_result)
            return text_result, score_result
    return None, [0, 0]


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id)
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """

    x1, y1, x2, y2, score, class_id = license_plate

    for i in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[i]

        if x1 > xcar1 and y1 > xcar1 and x2 < xcar2 and y2 < ycar2:
            return vehicle_track_ids[i]
    return -1, -1, -1, -1, -1
