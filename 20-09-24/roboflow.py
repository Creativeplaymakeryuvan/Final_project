
from ultralytics import YOLO
import cv2
import pytesseract
import re

model = YOLO('E:\\Transformers\\best.pt')

results = model.predict('E:\\Transformers\\images\\1029-receipt_jpg.rf.920e84b5652f273fa8971cade7222cd3.jpg', save=True, save_txt=True)

img = cv2.imread('E:\\Transformers\\images\\1029-receipt_jpg.rf.920e84b5652f273fa8971cade7222cd3.jpg')

for result in results:
    boxes = result.boxes.xyxy
    labels = result.boxes.cls
    confidences = result.boxes.conf
    names = result.names

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box) 
        label = names[int(labels[i])] 
        confidence = confidences[i] 


        roi = img[y1:y2, x1:x2]

        custom_config = r'--psm 6'
        text = pytesseract.image_to_string(roi, config=custom_config)

        text = text.strip()

        if label.lower() == "date":
            print(f'Extracted value for {label}: {text}')
        elif label.lower() == "totalprice":

            total_match = re.search(r'\$?\d+\.?\d*', text)
            extracted_value = total_match.group(0) if total_match else "No total found"
            print(f'Extracted value for {label}: {extracted_value}')
        elif label.lower() == "title":
            print(f'Extracted value for {label}: {text}')

cv2.imwrite('E:\\Transformers\\output\\processed_receipt.jpg', img)
