""" Module for preprocessing methods """

from tqdm import tqdm
import numpy as np
from utils import str_utils

PAD = 0


def get_neighbours(list_of_neighbours, vocabulary, n_neighbours):
    """Returns a list of neighbours and coordinates."""
    neighbours = list()
    neighbour_cords = list()
    
    for neighbour in list_of_neighbours:
        
        text = neighbour['text'].lower()
        if text not in vocabulary:
            if str_utils.is_number(text):
                neighbours.append(vocabulary['<NUMBER>'])
            else:
                neighbours.append(vocabulary['<RARE>'])
        else:
            neighbours.append(vocabulary[text])
        
        
        neighbour_cords.append(
            [
                neighbour['x'],
                neighbour['y']
            ]
        )
    
    len_neighbours = len(neighbours)
    if len_neighbours != n_neighbours:
        if len_neighbours > n_neighbours:
            neighbours = neighbours[:n_neighbours]
            neighbour_cords = neighbour_cords[:n_neighbours]
        else:
            neighbours.extend([vocabulary['<PAD>']] * (n_neighbours - len_neighbours))
            neighbour_cords.extend([[0., 0.]] * (n_neighbours - len_neighbours))

    return neighbours, neighbour_cords


def parse_input(annotations, fields_dict, n_neighbours=5, vocabulary=None):
    """Generates input samples from annotations data."""
    
    field_ids = list()
    candidate_cords = list()
    neighbours = list()
    neighbour_cords = list()
    labels = list()
    mask = list()
    n_classes = len(fields_dict)
    if not vocabulary:
        raise Exception("Vocabulary is missing. Use VocabularyBuilder to generate vocabulary of the data.")
    
    for annotation in tqdm(annotations, desc='Parsing Input'):
        
        fields = annotation['fields']
        
        for field in fields:
            if fields[field]['true_candidates']:
                _neighbours, _neighbour_cords = get_neighbours(
                    fields[field]['true_candidates'][0]['neighbours'],
                    vocabulary, n_neighbours
                )
                labels.append([1.])
                field_ids.append(np.eye(n_classes)[fields_dict[field]])
                candidate_cords.append(
                    [
                        fields[field]['true_candidates'][0]['x'],
                        fields[field]['true_candidates'][0]['y']
                    ]
                )
                neighbours.append(_neighbours)
                neighbour_cords.append(_neighbour_cords)
                mask.append([[1 if i else 0 for i in _neighbours]])

                for candidate in fields[field]['other_candidates']:

                    _neighbours, _neighbour_cords = get_neighbours(candidate['neighbours'], vocabulary, n_neighbours)
                    labels.append([0.])
                    field_ids.append(np.eye(n_classes)[fields_dict[field]])
                    candidate_cords.append(
                        [
                            candidate['x'],
                            candidate['y']
                        ]
                    )
                    neighbours.append(_neighbours)
                    neighbour_cords.append(_neighbour_cords)
                    mask.append([[1 if i else 0 for i in _neighbours]])

    return field_ids, candidate_cords, neighbours, neighbour_cords, mask, labels
