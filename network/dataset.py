from pathlib import Path

import torch
from torch.utils import data

from utils import xml_parser, Neighbour, visualizer, candidate
from utils import operations as op
from utils import preprocess

class DocumentsDataset(data.Dataset):
    """Stores the annotated documents dataset."""
    
    def __init__(self, xmls_path, ocr_path, image_path,candidate_path,
                 field_dict, n_neighbour=5, vocab=None):
        """ Initialize the dataset with preprocessing """
        annotation, classes_count, class_mapping = xml_parser.get_data(xmls_path)
        annotation = candidate.attach_candidate(annotation, candidate_path)
        annotation = Neighbour.attach_neighbour(annotation, ocr_path)
        annotation = op.normalize_positions(annotation)
        self.features, self.labels, self.vocab = preprocess.parse_input(annotation, field_dict, n_neighbour, vocab)        
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        
        return torch.tensor(self.features[idx]), self.labels[idx]
