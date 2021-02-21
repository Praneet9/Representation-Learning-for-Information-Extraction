from pytesseract import Output
import pytesseract
from glob import glob
import cv2
from tqdm import tqdm
import os
import json


def get_tesseract_results(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ocr_result = pytesseract.image_to_data(image, config='--oem 1 --psm 3', output_type=Output.DICT)
    return ocr_result


if __name__ == '__main__':

    dataset_dir = 'path/to/dataset/directory'
    images_dir = os.path.join(dataset_dir, 'images')
    tesseract_results = os.path.join(dataset_dir, 'tesseract_results_lstm')

    assert os.path.exists(images_dir), "images directory doesn't exist"
    if not os.path.exists(tesseract_results):
        os.makedirs(tesseract_results)

    images = glob(os.path.join(images_dir, '*.jpg'))

    for image in tqdm(images[:1], desc='Generating Tesseract Results'):
        image_name = os.path.splitext(os.path.split(image)[-1])[0]
        result = get_tesseract_results(image)
        with open(os.path.join(tesseract_results, image_name + '.json'), 'w') as f:
            json.dump(result, f)
