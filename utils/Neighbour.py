import json
import traceback
from utils import operations as op


def attach_neighbour(annotation, ocr_path):

    for anno in annotation:
        try:
            file_name = anno['filename']
            ocr_json = ocr_path / (file_name + ".json")
            with open(ocr_json, 'r') as f:
                ocr_data = json.load(f)

            empty_index = [i for i, ele in enumerate(ocr_data['text']) if ele == ""]
            for key in ocr_data.keys():
                ocr_data[key] = [j for i, j in enumerate(ocr_data[key]) if i not in empty_index]

            words = []
            for txt, x, y, w, h in zip(ocr_data['text'], ocr_data['left'], ocr_data['top'], ocr_data['width'],
                                       ocr_data['height']):
                x2 = x + w
                y2 = y + h

                words.append({'txt': txt, 'bbox': [x, y, x2, y2]})

            x_offset = int(anno['width'] * 0.1)
            y_offset = int(anno['height'] * 0.1)
            for a in anno['bboxes']:
                iou_scores = []
                for w in words:
                    iou_scores.append(op.bb_intersection_over_union([a['x1'], a['y1'], a['x2'], a['y2']], w['bbox']))
                if max(iou_scores) > 0.2:
                    max_ind = iou_scores.index(max(iou_scores))
                    a['keyword'] = words[max_ind]
                else:
                    print("No keyword found in OCR corresponding to: ", str(a), "filename :", file_name)
                    a['keyword'] = {}

                # neighbour

                a['neighbours'] = []

                neighbour_x1 = a['x1'] - x_offset
                neighbour_x1 = 1 if neighbour_x1 < 1 else neighbour_x1

                neighbour_y1 = a['y1'] - y_offset
                neighbour_y1 = 1 if neighbour_y1 < 1 else neighbour_y1

                neighbour_x2 = a['x2'] + x_offset
                neighbour_x2 = anno['width'] - 1 if neighbour_x2 >= anno['width'] else neighbour_x2

                neighbour_y2 = a['y2'] + y_offset
                neighbour_y2 = anno['height'] - 1 if neighbour_y2 >= anno['height'] else neighbour_y2

                neighbour_bbox = [neighbour_x1, neighbour_y1, neighbour_x2, neighbour_y2]
                iou_scores = []
                for w in words:
                    iou_scores.append(op.bb_intersection_over_boxB(neighbour_bbox, w['bbox']))

                for i, iou in enumerate(iou_scores):
                    if iou > 0.2:
                        a['neighbours'].append(words[i])

        except Exception:
            trace = traceback.format_exc()
            print("Error in finding neighbour: %s : %s" % (anno['filename'], trace))

    return annotation