from utils import config
import extract_candidates
import re
import json
import os
from dateparser.search import search_dates
import generate_tesseract_results

candidates_folder=str(config.CANDIDATE_DIR)+'/'
images_folder=str(config.IMAGE_DIR)+'/'



if __name__ == '__main__':
    print("main")
    for image in os.listdir(images_folder):
        print(image)
        path=images_folder+image
        ocr_data=generate_tesseract_results.get_tesseract_results(path)
        candidates=extract_candidates.get_candidates(ocr_data)
        with open(candidates_folder+image.replace('.jpg','.json'),'w') as f:
            json.dump(candidates,f)

