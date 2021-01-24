import torch
from torch.utils import data

from utils import xml_parser, Neighbour, candidate
from utils import operations as op
from utils import preprocess


class DocumentsDataset(data.Dataset):
    """Stores the annotated documents dataset."""
    
    def __init__(self, xmls_path, ocr_path, image_path, candidate_path, n_neighbour=5, vocab=None):
        """ Initialize the dataset with preprocessing """
        annotation, classes_count, class_mapping = xml_parser.get_data(xmls_path)
        print("Class Mapping:", class_mapping)
        print("Classs counts:", classes_count)
        annotation = candidate.attach_candidate(annotation, candidate_path)
        annotation = Neighbour.attach_neighbour(annotation, ocr_path)
        annotation = op.normalize_positions(annotation)
        _data = preprocess.parse_input(annotation, class_mapping, n_neighbour, vocab)
        self.field_ids, self.candidate_cords, self.neighbours, self.neighbour_cords, self.labels, self.vocab = _data
    
    def __len__(self):
        return len(self.field_ids)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.field_ids[idx]).type(torch.FloatTensor),
            torch.tensor(self.candidate_cords[idx]).type(torch.FloatTensor),
            torch.tensor(self.neighbours[idx]),
            torch.tensor(self.neighbour_cords[idx]).type(torch.FloatTensor),
            torch.tensor(self.labels[idx]).type(torch.FloatTensor)
        )
