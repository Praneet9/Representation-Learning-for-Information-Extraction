from pathlib import Path
from utils import xml_parser,Neighbour, visualizer

this_dir = Path.cwd()
xmls_path = this_dir / "dataset" / "xmls"
ocr_path = this_dir / "dataset" / "tesseract_results"
image_path = this_dir / "dataset" / "images"

annotation, classes_count, class_mapping = xml_parser.get_data(xmls_path)

images = list(image_path.glob("*.jpg"))

for img in images:
    visualizer.tesseract_ocr(img, ocr_path / (img.stem + ".json"))

#annotation = Neighbour.attach_neighbour(annotation, ocr_path)

print()

