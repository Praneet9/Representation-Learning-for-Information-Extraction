from pathlib import Path

import cv2
from torch.utils import data

from utils import xml_parser, Neighbour, visualizer, candidate
from utils import operations as op
from network import dataset

this_dir = Path.cwd()
xmls_path = this_dir / "dataset" / "xmls"
ocr_path = this_dir / "dataset" / "tesseract_results"
image_path = this_dir / "dataset" / "images"
candidate_path = this_dir / "dataset" / "candidates"

# annotation, classes_count, class_mapping = xml_parser.get_data(xmls_path)
# annotation = candidate.attach_candidate(annotation, candidate_path)
# annotation = Neighbour.attach_neighbour(annotation, ocr_path)
# annotation = op.normalize_positions(annotation)

# images = list(image_path.glob("*.jpg"))
# for img in images:
#     out_img = visualizer.tesseract_ocr(img, ocr_path / (img.stem + ".json"))
#     cv2.imwrite('test.jpg', out_img)
#     break

# print(annotation)

batch_size = 32
# field_dict = {'invoice_date':0, 'invoice_no':1, 'total':2}

doc_dataset = dataset.DocumentsDataset(xmls_path, ocr_path, image_path, candidate_path)

dataloader = data.DataLoader(doc_dataset, batch_size=32, shuffle=True)

field_id, candidate_pos, neighbour, neighbour_pos, label = next(iter(dataloader))

print("s")