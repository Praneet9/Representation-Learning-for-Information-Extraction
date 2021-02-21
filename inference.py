import torch
import cv2
from utils import Neighbour, config, preprocess, generate_tesseract_results
import extract_candidates
import pickle
import traceback
import numpy as np
import argparse
import os


def attach_neighbour_candidates(width, height, ocr_data, candidates):
    empty_index = [i for i, ele in enumerate(ocr_data['text']) if ele == ""]
    for key in ocr_data.keys():
        ocr_data[key] = [j for i, j in enumerate(ocr_data[key]) if i not in empty_index]
    words = []
    for txt, x, y, w, h in zip(ocr_data['text'], ocr_data['left'], ocr_data['top'], ocr_data['width'],
                               ocr_data['height']):
        x2 = x + w
        y2 = y + h
        words.append({'text': txt, 'x1': x, 'y1': y, 'x2': x2, 'y2': y2})
    x_offset = int(width * 0.1)
    y_offset = int(height * 0.1)
    for cls, both_cads in candidates.items():
        for cad in both_cads:
            neighbours = Neighbour.find_neighbour(cad, words, x_offset, y_offset, width, height)
            cad['neighbours'] = neighbours
    return candidates


def load_saved_vocab(path):
    cached_data = pickle.load(open(path, 'rb'))
    return cached_data['vocab'], cached_data['mapping']


def parse_input(annotations, fields_dict, n_neighbours=5, vocabulary=None):
    """Generates input samples from annotations data."""
    field_ids = list()
    candidate_cords = list()
    neighbours = list()
    neighbour_cords = list()
    n_classes = len(fields_dict)
    for field, value in annotations.items():
        if annotations[field]:
            for idx, val in enumerate(value):
                _neighbours, _neighbour_cords = preprocess.get_neighbours(
                    val['neighbours'],
                    vocabulary, n_neighbours
                )
                field_ids.append(np.eye(n_classes)[fields_dict[field]])
                candidate_cords.append(
                    [
                        val['x'],
                        val['y']
                    ]
                )
                neighbours.append(_neighbours)
                neighbour_cords.append(_neighbour_cords)
    return torch.Tensor(field_ids).type(torch.FloatTensor), torch.Tensor(candidate_cords).type(
        torch.FloatTensor), torch.Tensor(neighbours).type(torch.int64), torch.Tensor(neighbour_cords).type(
        torch.FloatTensor)


def normalize_coordinates(annotations, width, height):
    try:
        for cls, cads in annotations.items():
            for i, cd in enumerate(cads):
                cd = cd.copy()
                x1 = cd['x1']
                y1 = cd['y1']
                x2 = cd['x2']
                y2 = cd['y2']
                cd['x'] = ((x1 + x2) / 2) / width
                cd['y'] = ((y1 + y2) / 2) / height
                neighbours = []
                for neh in cd['neighbours']:
                    neh = neh.copy()
                    x1_neh = neh['x1']
                    y1_neh = neh['y1']
                    x2_neh = neh['x2']
                    y2_neh = neh['y2']
                    # calculating neighbour position w.r.t candidate
                    neh['x'] = (((x1_neh + x2_neh) / 2) / width) - cd['x']
                    neh['y'] = (((y1_neh + y2_neh) / 2) / height) - cd['y']
                    neighbours.append(neh)
                cd['neighbours'] = neighbours
                annotations[cls][i] = cd
    except Exception:
        trace = traceback.format_exc()
        print("Error in normalizing position: %s : %s" % (trace, trace))
    return annotations


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Inference outputs')
    parser.add_argument('--cached_pickle', dest='saved_path',
                        help='Enter the path of the saved pickle during training',
                        default='cached_data.pickle', type=str)
    parser.add_argument('--load_saved_model', dest='load_model',
                        help='directory to load models', default="model.pth",
                        type=str)
    parser.add_argument('--image', dest='image_path',
                        help='directory to load models',
                        type=str)
    parser.add_argument('--visualize', dest='visualize',
                        help='directory to load models',
                        action='store_true')
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if not os.path.exists(args.image_path):
        raise Exception("Image not found")
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    image = cv2.imread(args.image_path)
    height, width, _ = image.shape
    ocr_results = generate_tesseract_results.get_tesseract_results(image)
    vocab, class_mapping = load_saved_vocab(args.saved_path)
    candidates = extract_candidates.get_candidates(ocr_results)
    candidates_with_neighbours = attach_neighbour_candidates(width, height, ocr_results, candidates)
    annotation = normalize_coordinates(candidates_with_neighbours, width, height)
    _data = parse_input(annotation, class_mapping, config.NEIGHBOURS, vocab)
    field_ids, candidate_cords, neighbours, neighbour_cords = _data
    rlie = torch.load(args.load_model)
    rlie = rlie.to(device)
    field_ids = field_ids.to(device)
    candidate_cords = candidate_cords.to(device)
    neighbours = neighbours.to(device)
    neighbour_cords = neighbour_cords.to(device)
    field_idx_candidate = np.argmax(field_ids.detach().to('cpu').numpy(), axis=1)
    with torch.no_grad():
        rlie.eval()
        val_outputs = rlie(field_ids, candidate_cords, neighbours, neighbour_cords)
    val_outputs = val_outputs.to('cpu').numpy()
    out = {cl: np.argmax(val_outputs[np.where(field_idx_candidate == cl)]) for cl in np.unique(field_idx_candidate)}
    true_candidate_color = (0, 255, 0)
    output_candidates = {}
    output_image = image.copy()
    for idx, (key, value) in enumerate(candidates.items()):
        if idx in out:
            candidate_idx = out[idx]
            cand = candidates[key][candidate_idx]
            output_candidates[key] = cand['text']
            cand_coords = [cand['x1'], cand['y1'], cand['x2'], cand['y2']]
            cv2.rectangle(output_image, (cand_coords[0], cand_coords[1]), (cand_coords[2], cand_coords[3]),
                          true_candidate_color, 5)
    if args.visualize:
        cv2.imshow('Visualize', output_image)
        cv2.waitKey(0)
    return output_candidates


if __name__ == '__main__':
    main()