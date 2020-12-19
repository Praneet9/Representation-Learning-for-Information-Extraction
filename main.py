from pathlib import Path
from utils import xml_parser, Neighbour, visualizer, candidate
from utils import operations as op
import cv2

this_dir = Path.cwd()
xmls_path = this_dir / "dataset" / "xmls"
ocr_path = this_dir / "dataset" / "tesseract_results"
image_path = this_dir / "dataset" / "images"
candidate_path = this_dir / "dataset" / "candidates"

annotation, classes_count, class_mapping = xml_parser.get_data(xmls_path)
annotation = candidate.attach_candidate(annotation, candidate_path)
annotation = Neighbour.attach_neighbour(annotation, ocr_path)
annotation = op.normalize_positions(annotation)

# images = list(image_path.glob("*.jpg"))
# for img in images:
#     out_img = visualizer.tesseract_ocr(img, ocr_path / (img.stem + ".json"))
#     cv2.imwrite('test.jpg', out_img)
#     break

print()

