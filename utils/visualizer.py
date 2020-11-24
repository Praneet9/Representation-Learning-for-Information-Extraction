import json
import cv2

def tesseract_ocr(image_path, ocr_path):

    image = cv2.imread(image_path.as_posix())

    with open(ocr_path, 'r') as f:
        ocr_data = json.load(f)

    empty_index = [i for i, ele in enumerate(ocr_data) if ele == ""]
    print()
