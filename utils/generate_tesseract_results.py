from pytesseract import Output
from pathlib import Path
import cv2
import json
import pytesseract
imgs = list(Path('ALL').glob("*.jpg"))
for img in imgs:
    try:
        image = cv2.imread(str(img))
        d = pytesseract.image_to_data(image, output_type=Output.DICT)
        with open('tesseract_results/'+str(img.stem) +'.json','w') as f:
            json.dump(d,f)
    except Exception as e:
        print(img,e)
        continue