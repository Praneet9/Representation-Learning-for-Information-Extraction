import xml.etree.ElementTree as ET
from pathlib import Path
import traceback


def get_data(xml_path):

    annotations = []
    classes_count = {}
    class_mapping = {}

    annotation_files = xml_path.glob("*.xml")
    for annot in annotation_files:
        try:
            et = ET.parse(annot)
            element = et.getroot()

            element_objs = element.findall('object')
            element_width = int(element.find('size').find('width').text)
            element_height = int(element.find('size').find('height').text)

            if len(element_objs) < 1:
                continue

            annotation_data = {'filename': annot.stem, 'width': element_width,
                               'height': element_height, 'field': {'invoice_no': {'true_candidate': [],
                                                                                  'other_candidate': []},
                                                                   'invoice_date': {'true_candidate': [],
                                                                                    'other_candidate': []},
                                                                   'total': {'true_candidate': [],
                                                                             'other_candidate': []}}}
            for i, cls in enumerate(annotation_data['field']):
                classes_count[cls] = 0
                class_mapping[cls] = i

            for element_obj in element_objs:
                class_name = element_obj.find('name').text
                if class_name not in annotation_data['field']:
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
                annotation_data['field'][class_name]['true_candidate'].append({'bbox': {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2},
                                                             'difficult': difficulty})

            annotations.append(annotation_data)

        except Exception:
            print(traceback.format_exc())

    return annotations, classes_count, class_mapping


if __name__ == '__main__':
    this_dir = Path.cwd()
    k, l, m = get_data( this_dir.parent / "dataset")
    print(len(k), l, m)