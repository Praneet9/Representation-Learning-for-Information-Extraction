""" Module to for preprocessing methods """
import json
from tqdm import tqdm 

PAD = 0

def get_neighbours(list_of_neighbours, vocabulary, n_neighbours):
    """Returns a list of neighbours and coordinates."""
    neighbours = list()
    neighbour_cords = list()
    
    for neighbour in list_of_neighbours:
        if neighbour['text'] not in vocabulary:
            vocabulary[neighbour['text']] = len(vocabulary)

        neighbours.append(vocabulary[neighbour['text']])
        neighbour_cords.append(
            [
                neighbour['x'],
                neighbour['y']
            ]
        )
    
    len_neighbours = len(neighbours)
    if len_neighbours != n_neighbours:
        if  len_neighbours > n_neighbours:
            neighbours = neighbours[:n_neighbours]
            neighbour_cords = neighbour_cords[:n_neighbours]
        else:
            neighbours.append(vocabulary['<PAD>'])
            neighbour_cords.extend([[0., 0.]] * (n_neighbours - len_neighbours))

    return neighbours, neighbour_cords

def parse_input(annotations, fields_dict, n_neighbours=5, vocabulary=None):
    """Generates input samples from annotations data."""
    
    field_ids = list()
    candidate_cords = list()
    neighbours = list()
    neighbour_cords = list()
    labels = list()
    if not vocabulary:
        vocabulary = { '<PAD>':PAD }

    for annotation in tqdm(annotations, desc='Parsing Input'):
        
        fields = annotation['fields']
        
        for field in fields:
            if fields[field]['true_candidates']:
                _neighbours, _neighbour_cords = get_neighbours(
                    fields[field]['true_candidates'][0]['neighbours'],
                    vocabulary, n_neighbours
                )
                labels.append(1.)
                field_ids.append(fields_dict[field])
                candidate_cords.append(
                    [
                        fields[field]['true_candidates'][0]['x'],
                        fields[field]['true_candidates'][0]['y']
                    ]
                )
                neighbours.append(_neighbours)
                neighbour_cords.append(_neighbour_cords)
               
                for candidate in fields[field]['other_candidates']:

                    _neighbours, _neighbour_cords = get_neighbours(candidate['neighbours'], vocabulary, n_neighbours)
                    labels.append(0.)
                    field_ids.append(fields_dict[field])
                    candidate_cords.append(
                        [
                            candidate['x'],
                            candidate['y']
                        ]
                    )
                    neighbours.append(_neighbours)
                    neighbour_cords.append(_neighbour_cords)
                    
                    
    return field_ids, candidate_cords, neighbours, neighbour_cords, labels, vocabulary