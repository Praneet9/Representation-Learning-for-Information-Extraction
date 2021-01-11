""" Module to for preprocessing methods """
import json
from tqdm import tqdm 

PAD = 0

def get_neighbours(list_of_neighbours, vocabulary, n_neighbours):
    """Returns a list of neighbours and coordinates."""
    neighbours = list()
    
    for neighbour in list_of_neighbours:
        if neighbour['text'] not in vocabulary:
            vocabulary[neighbour['text']] = len(vocabulary)

        neighbours.extend(
            [
                vocabulary[neighbour['text']],
                neighbour['x'],
                neighbour['y']
            ]
        )
    
    len_neighbours = int(len(neighbours) / 3) 
    if len_neighbours != n_neighbours:
        if  len_neighbours > n_neighbours:
            neighbours = neighbours[:(n_neighbours * 3)]
        else:
            neighbours.extend(['<PAD>', 0., 0.] * (n_neighbours - len_neighbours))

    return neighbours

def parse_input(annotations, fields_dict, n_neighbours=5, vocabulary=None):
    """Generates input samples from annotations data."""

    x = list()
    Y = list()
    if not vocabulary:
        vocabulary = { '<PAD>':PAD }

    for annotation in tqdm(annotations, desc='Parsing Input'):
        
        fields = annotation['fields']
        
        for field in fields:
            if fields[field]['true_candidates']:
                Y.append(1.)
                neighbours = get_neighbours(
                    fields[field]['true_candidates'][0]['neighbours'],
                    vocabulary, n_neighbours
                )
                x.append(
                    [
                        fields_dict[field],
                        fields[field]['true_candidates'][0]['x'],
                        fields[field]['true_candidates'][0]['y']
                    ] + neighbours)

                for candidate in fields[field]['other_candidates']:

                    Y.append(0.)
                    neighbours = get_neighbours(candidate['neighbours'], vocabulary, n_neighbours)
                    x.append(
                        [
                            fields_dict[field],
                            candidate['x'],
                            candidate['y'],
                        ] + neighbours)

    return x, Y, vocabulary