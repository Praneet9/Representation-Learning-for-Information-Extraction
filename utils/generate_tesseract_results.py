from pytesseract import Output
from pathlib import Path
import cv2
import json
import pytesseract
import os
from tqdm import tqdm


imgs = list(Path('../dataset/ALL').glob("*.jpg"))
for img in tqdm(imgs[950:]):
    try:
        if os.path.exists('../dataset/tesseract_results/'+str(img.stem) + '.json'):
            continue
        image = cv2.imread(str(img))
        d = pytesseract.image_to_data(image, config='--oem 1 --psm 3', output_type=Output.DICT)
        with open('../dataset/tesseract_results/'+str(img.stem) + '.json', 'w') as f:
            json.dump(d, f)
    except Exception as e:
        print(img,e)
        continue
