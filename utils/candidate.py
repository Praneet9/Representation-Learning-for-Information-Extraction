import json
import traceback
from utils import operations as op


def attach_candidate(annotation, candidate_path):

    for anno in annotation:
        try:
            file_name = anno['filename']
            candidate_json = candidate_path / (file_name + ".json")
            with open(candidate_json, 'r') as f:
                candidate_data = json.load(f)

            for cls, cads in candidate_data.items():
                if cls == 'invoice_number':
                    clss = 'invoice_no'
                else:
                    clss = cls

                for true_cad in anno['fields'][clss]['true_candidates']:
                    for cad in cads:
                        iou = op.bb_intersection_over_union([true_cad['x1'], true_cad['y1'],
                                                             true_cad['x2'], true_cad['y2']],
                                                            [cad['x1'], cad['y1'], cad['x2'], cad['y2']])
                        if iou > 0.5:
                            anno['fields'][clss]['true_candidates'].remove(true_cad)
                            anno['fields'][clss]['true_candidates'].append(cad)
                            candidate_data[cls].remove(cad)
                            break
                anno['fields'][clss]['other_candidates'] = candidate_data[cls]

        except Exception:
            trace = traceback.format_exc()
            print("Error in processing candidate: %s : %s" % (anno['filename'], trace))

    return annotation
