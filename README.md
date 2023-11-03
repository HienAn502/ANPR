# ANPR
Recommend clone project with Pycharm or similar, 
or else recommend create virtual environment.
1. Clone this repository:
```commandline
git clone https://github.com/HienAn502/ANPR.git
git checkout master 
```
Make sure the working directory: X:xxxx/xxxx/ANPR
2. Install required libraries:
```commandline
pip install -r "requirements.txt" 
```
3. Clone the required repository for project:
```commandline
git clone https://github.com/abewley/sort.git
```
4. Dataset from Roboflow

https://universe.roboflow.com/hai-binh-nguyen-7za5j/lp-vietnam/dataset/1
Download with YOLOv8 format and unzip inside folder ANPR

# Code

**license_plate_model.py**

- Add this line at the top of file dataset/data.yaml
- Change xxxx by your directory

```commandline
path: X:\xxxx\ANPR\dataset
```
- Code for training model with file dataset/data.yaml
- You can train with other dataset but remember to format in YOLOv8

**main.py**

- Using 2 models: coco model for vehicle detection and our model for license plate detection
- Applying on video sample.mp4 and recognize license number using OCR (easyocr)

**util.py**

- Method: write_csv(results, output_path)
```text
Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
```
- Method: license_complies_format(text):

*Only apply for motorcycle, will extend to be able to apply for car in the future*
```text
Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text

    Returns:
        bool: True if the license plate complies with the format, False otherwise
```
- Method: format_license(text):
```text
Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text

    Returns:
        str: Formatted license plate text
```
- Method: read_license_plate(license_plate_crop)
```text
Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
```
- Method: get_car(license_plate, vehicle_track_ids):
```text
Retrieve the vehicle coordinates and ID based on the license plate coordinates

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id)
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
```
# Code flow:

- Cut frame from video using opencv
- Detect vehicle from frame using coco_model
- Using sort.mot_tracker to track vehicle
- Detect license plate from frame using license_plate_model
- Assign detected license plate to correct vehicle
- Process license plate cut, so it's easier for easyocr to read
- Read license plate number
- Process final result and write them to result.csv