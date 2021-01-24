from tqdm import tqdm
import traceback

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def bb_intersection_over_boxB(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    # boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxBArea)

    # return the intersection over union value
    return iou


def normalize_positions(annotations):

    candidate_types = ['true_candidates', 'other_candidates']
    for anno in tqdm(annotations, desc="normalizing position coordinates"):
        try:
            for cls, cads in anno['fields'].items():
                for cd_typ in candidate_types:
                    for i, cd in enumerate(cads[cd_typ]):
                        cd = cd.copy()
                        x1 = cd['x1']
                        y1 = cd['y1']
                        x2 = cd['x2']
                        y2 = cd['y2']
                        cd['x'] = ((x1 + x2) / 2) / anno['width']
                        cd['y'] = ((y1 + y2) / 2) / anno['height']
                        neighbours = []
                        for neh in cd['neighbours']:
                            neh = neh.copy()
                            x1_neh = neh['x1']
                            y1_neh = neh['y1']
                            x2_neh = neh['x2']
                            y2_neh = neh['y2']

                            # calculating neighbour position w.r.t candidate
                            neh['x'] = (((x1_neh + x2_neh) / 2) / anno['width']) - cd['x']
                            neh['y'] = (((y1_neh + y2_neh) / 2) / anno['height']) - cd['y']
                            neighbours.append(neh)
                        cd['neighbours'] = neighbours
                        anno['fields'][cls][cd_typ][i] = cd
        except Exception:
            trace = traceback.format_exc()
            print("Error in normalizing position: %s : %s" % (anno['filename'], trace))
            break

    return annotations