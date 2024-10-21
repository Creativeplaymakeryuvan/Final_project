from flask import Flask, jsonify, request
from ultralytics import YOLO
import cv2
import pytesseract
import re
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = YOLO('E:\\Python-server\\best(2).pt')

@app.route('/extract-data', methods=['POST'])
def extract_data():
    if 'image' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({"error": "No selected file."}), 400
    
    image_path = os.path.join('uploads', image_file.filename)
    image_file.save(image_path)

    results = model.predict(image_path, save=True, save_txt=True)

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
                total_match = re.search(r'\$?(\d+\.?\d*)', text)
                extracted_data['totalprice'] = total_match.group(1) if total_match else "No total found"
            elif label.lower() == "title":
                extracted_data['title'] = text

    os.remove(image_path)

    return jsonify({
        "extracted_data": extracted_data
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
