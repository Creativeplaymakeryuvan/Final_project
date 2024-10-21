from ultralytics import YOLO
from datetime import datetime
import cv2
import pytesseract
import re
import os

# Load the YOLO model
model = YOLO('E:\\Python-server\\best(2).pt')

def extract_data(image_path):
    # Check if the image file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError("The specified image file does not exist.")
    
    # Use YOLO model to predict on the image
    results = model.predict(image_path, save=True, save_txt=True)

    # Read the image using OpenCV
    img = cv2.imread(image_path)
    extracted_data = {}

    for result in results:
        boxes = result.boxes.xyxy
        labels = result.boxes.cls
        confidences = result.boxes.conf
        names = result.names

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = names[int(labels[i])]

            roi = img[y1:y2, x1:x2]
            custom_config = r'--psm 6'
            text = pytesseract.image_to_string(roi, config=custom_config).strip()

            if label.lower() == "date":
                extracted_data['date'] = text
            elif label.lower() == "totalprice":
                  total_match = re.search(r'\$?(\d+\.?\d*)', text)  # Capture the number only
                  extracted_data['totalprice'] = total_match.group(1) if total_match else "No total found"

            elif label.lower() == "title":
                extracted_data['title'] = text

    return extracted_data

if __name__ == '__main__':
    image_file_path = 'E:\\Python-server\\images\\1000-receipt_jpg.rf.6135dc2e5cbeb3378a627ed420ec94f9.jpg'
    try:
        extracted_data = extract_data(image_file_path)
        print("Extracted Data:", extracted_data)
    except Exception as e:
        print("Error:", str(e))
