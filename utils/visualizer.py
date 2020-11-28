import json
import cv2

def tesseract_ocr(image_path, ocr_path):

    image = cv2.imread(image_path.as_posix())

    with open(ocr_path, 'r') as f:
        ocr_data = json.load(f)

    empty_index = [i for i, ele in enumerate(ocr_data['text']) if ele == ""]
    for key in ocr_data.keys():
        ocr_data[key] = [j for i, j in enumerate(ocr_data[key]) if i not in empty_index]

    for txt, x, y, w, h in zip(ocr_data['text'], ocr_data['left'], ocr_data['top'], ocr_data['width'], ocr_data['height']):
        x2 = x + w
        y2 = y + h
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 1)
        cv2.putText(image, txt, (x,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

    return image

