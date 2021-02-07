import xml.etree.ElementTree as ET
from pathlib import Path
import traceback
from tqdm import tqdm
from utils import config


def get_data(xml_path, split_name='train'):

    annotations = []
    classes_count = {}
    class_mapping = {}

    split_file_path = config.SPLIT_DIR / f"{split_name}.txt"
    with open(split_file_path, 'r') as f:
        valid_files = f.read().split("\n")
    annotation_files = list(xml_path.glob("*.xml"))
    annotation_files = [an for an in annotation_files if an.stem in valid_files]
    for annot in tqdm(annotation_files, desc="Reading Annotations"):
        try:
            et = ET.parse(annot)
            element = et.getroot()

            element_objs = element.findall('object')
            element_width = int(element.find('size').find('width').text)
            element_height = int(element.find('size').find('height').text)

            if len(element_objs) < 1:
                continue

            annotation_data = {'filename': annot.stem, 'width': element_width,
                               'height': element_height, 'fields': {'invoice_no': {'true_candidates': [],
                                                                                   'other_candidates': []},
                                                                    'invoice_date': {'true_candidates': [],
                                                                                     'other_candidates': []},
                                                                    'total': {'true_candidates': [],
                                                                              'other_candidates': []}}}
            for i, cls in enumerate(annotation_data['fields']):
                if cls not in classes_count:
                    classes_count[cls] = 0
                    class_mapping[cls] = i

            for element_obj in element_objs:
                class_name = element_obj.find('name').text
                if class_name not in annotation_data['fields']:
                    print("Unidentified field Found:", class_name, "in file:", annot.name)
                    continue
                else:
                    classes_count[class_name] += 1

                obj_bbox = element_obj.find('bndbox')
                x1 = int(round(float(obj_bbox.find('xmin').text)))
                y1 = int(round(float(obj_bbox.find('ymin').text)))
                x2 = int(round(float(obj_bbox.find('xmax').text)))
                y2 = int(round(float(obj_bbox.find('ymax').text)))
                difficulty = int(element_obj.find('difficult').text) == 1
                annotation_data['fields'][class_name]['true_candidates'].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2,
                                                                                 'difficult': difficulty})

            annotations.append(annotation_data)

        except Exception:
            print(traceback.format_exc())

    return annotations, classes_count, class_mapping


if __name__ == '__main__':
    this_dir = Path.cwd()
    k, l, m = get_data( this_dir.parent / "dataset")
    print(len(k), l, m)