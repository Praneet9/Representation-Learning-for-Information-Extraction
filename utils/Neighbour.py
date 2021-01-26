import json
import traceback
from tqdm import tqdm

from utils import operations as op
from utils import vocabulary

def find_neighbour(cad, words, x_offset, y_offset, width, height):
    # iou_scores = []
    # for w in words:
    #     iou_scores.append(op.bb_intersection_over_union([cad['x1'], cad['y1'], cad['x2'], cad['y2']],
    #                                                     [w['x1'], w['y1'], w['x2'], w['y2']]))

    # if max(iou_scores) > 0.2:
    #     max_ind = iou_scores.index(max(iou_scores))
    #     a['keyword'] = words[max_ind]
    # else:
    #     print("No keyword found in OCR corresponding to: ", str(a), "filename :", file_name)
    #     a['keyword'] = {}

    # neighbour
    words_copy = words.copy()
    if cad in words_copy:
        words_copy.remove(cad)
    neighbours = []

    neighbour_x1 = cad['x1'] - x_offset
    neighbour_x1 = 1 if neighbour_x1 < 1 else neighbour_x1

    neighbour_y1 = cad['y1'] - y_offset
    neighbour_y1 = 1 if neighbour_y1 < 1 else neighbour_y1

    neighbour_x2 = cad['x2'] + x_offset
    neighbour_x2 = width - 1 if neighbour_x2 >= width else neighbour_x2

    neighbour_y2 = cad['y2'] + y_offset
    neighbour_y2 = height - 1 if neighbour_y2 >= height else neighbour_y2

    neighbour_bbox = [neighbour_x1, neighbour_y1, neighbour_x2, neighbour_y2]
    iou_scores = []
    for w in words_copy:
        iou_scores.append(op.bb_intersection_over_boxB(neighbour_bbox, [w['x1'], w['y1'], w['x2'], w['y2']]))

    for i, iou in enumerate(iou_scores):
        if iou > 0.5:
            neighbours.append(words_copy[i])

    return neighbours


def attach_neighbour(annotation, ocr_path, vocab_size):
    
    vocab_builder = vocabulary.VocabularyBuilder(max_size=vocab_size)
    
    for anno in tqdm(annotation, desc="Attaching Neighbours"):
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

                words.append({'text': txt, 'x1': x, 'y1': y, 'x2': x2, 'y2': y2})
                
                vocab_builder.add(txt)

            x_offset = int(anno['width'] * 0.1)
            y_offset = int(anno['height'] * 0.1)
            for cls, both_cads in anno['fields'].items():
                for cad in both_cads['true_candidates']:
                    neighbours = find_neighbour(cad, words, x_offset, y_offset, anno['width'], anno['height'])
                    cad['neighbours'] = neighbours
                for cad in both_cads['other_candidates']:
                    neighbours = find_neighbour(cad, words, x_offset, y_offset, anno['width'], anno['height'])
                    cad['neighbours'] = neighbours

        except Exception:
            trace = traceback.format_exc()
            print("Error in finding neighbour: %s : %s" % (anno['filename'], trace))
            break
            
    _vocab = vocab_builder.build()

    return annotation, _vocab