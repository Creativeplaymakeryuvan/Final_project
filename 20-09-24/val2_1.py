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
            # print(f'Extracted text for {label}: {text}')
            total_match = re.search(r'\$?\d+\.?\d*', text)
            extracted_value = total_match.group(0) if total_match else "No total found"
            print(f'Extracted value for {label}: {extracted_value}')
        elif label.lower() == "title":
            extracted_value = text
            print(f'Extracted value for {label}: {extracted_value}')

        





# # 3

# from ultralytics import YOLO
# import cv2
# import easyocr
# import pytesseract
# import json

# model = YOLO('E:\\Transformers\\best.pt')

# results = model.predict('E:\\Transformers\\images\\1000-receipt_jpg.rf.6135dc2e5cbeb3378a627ed420ec94f9.jpg', save=True, save_txt=True)

# reader = easyocr.Reader(['en'])

# extracted_data = []

# image_path = 'E:\\Transformers\\images\\1000-receipt_jpg.rf.6135dc2e5cbeb3378a627ed420ec94f9.jpg'
# img = cv2.imread(image_path)

# if img is None:
#     print("Image not found or could not be loaded.")
# else:
#     for result in results:
#         boxes = result.boxes.xyxy  
#         labels = result.boxes.cls
#         confidences = result.boxes.conf  
#         names = result.names

#         print(f"Detected {len(boxes)} boxes with labels: {labels}")

#         predictions = []
#         for i, box in enumerate(boxes):
#             x1, y1, x2, y2 = map(int, box)
#             label = names[int(labels[i])]
#             confidence = confidences[i] 
#             predictions.append((x1, y1, x2, y2, label, confidence))

#         for pred in predictions:
#             x1, y1, x2, y2, label, confidence = pred

#             roi = img[y1:y2, x1:x2]

#             custom_config = r'--oem 3 --psm 6'
#             ocr_result = pytesseract.image_to_string(roi, config=custom_config)

#             print(f"Extracted text for {label}: {ocr_result}")

#             extracted_data.append({
#                 'label': label,
#                 'confidence': float(confidence),
#                 'extracted_text': ocr_result.strip()
#             })

#     print(json.dumps(extracted_data, indent=4))



# 3

# from ultralytics import YOLO
# import cv2
# import pytesseract

# model = YOLO('E:\\Transformers\\best.pt')

# results = model.predict('E:\\Transformers\\images\\1000-receipt_jpg.rf.6135dc2e5cbeb3378a627ed420ec94f9.jpg', save=True, save_txt=True)
    
# for result in results:
#     boxes = result.boxes.xyxy  
#     labels = result.boxes.cls 
#     confidences = result.boxes.conf 
#     names = result.names

#     for i, box in enumerate(boxes):
#         x1, y1, x2, y2 = map(int, box)
#         label = names[int(labels[i])]
#         confidence = confidences[i]

#         print(f'Box: {box}, Label: {label}, Confidence: {confidence:.2f}')


#         img = cv2.imread('E:\\Transformers\\images\\1000-receipt_jpg.rf.6135dc2e5cbeb3378a627ed420ec94f9.jpg')

#         cropped_img = img[y1:y2, x1:x2]
#         cv2.imwrite(f'cropped_{label}.png', cropped_img)

#         text = pytesseract.image_to_string(cropped_img)

#         print(f'Extracted text for {label}: {text}')

