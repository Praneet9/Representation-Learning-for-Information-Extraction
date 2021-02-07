import torch
from torch.utils import data
from utils import config
from utils import xml_parser, Neighbour, candidate
from utils import operations as op
from utils import preprocess
import pickle


class DocumentsDataset(data.Dataset):
    """Stores the annotated documents dataset."""
    
    def __init__(self, split_name='train'):
        """ Initialize the dataset with preprocessing """
        cached_data_path = config.OUTPUT_DIR / f"cached_data_{split_name}.pickle"
        if cached_data_path.exists():
            print("Preprocessed data available, Loading data from cache...")
            with open(cached_data_path, "rb") as f:
                cached_data = pickle.load(f)
            classes_count = cached_data['count']
            class_mapping = cached_data['mapping']
            print("\nClass Mapping:", class_mapping)
            print("Classs counts:", classes_count)
            _data = cached_data['data']
            self.vocab = cached_data['vocab']
            self.field_ids, self.candidate_cords, self.neighbours, self.neighbour_cords, self.mask, self.labels = _data
        else:
            print("Preprocessed data not available")
            annotation, classes_count, class_mapping = xml_parser.get_data(config.XML_DIR, split_name)
            print("Class Mapping:", class_mapping)
            print("Classs counts:", classes_count)
            annotation = candidate.attach_candidate(annotation, config.CANDIDATE_DIR)
            annotation, self.vocab = Neighbour.attach_neighbour(annotation, config.OCR_DIR, vocab_size=config.VOCAB_SIZE)
            annotation = op.normalize_positions(annotation)
            _data = preprocess.parse_input(annotation, class_mapping, config.NEIGHBOURS, self.vocab)
            self.field_ids, self.candidate_cords, self.neighbours, self.neighbour_cords, self.mask, self.labels = _data
            cached_data = {'count': classes_count, "mapping": class_mapping, 'vocab': self.vocab, 'data': _data}
            print("Saving Cache..")
            with open(cached_data_path, 'wb') as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Done !!")
    
    def __len__(self):
        return len(self.field_ids)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.field_ids[idx]).type(torch.FloatTensor),
            torch.tensor(self.candidate_cords[idx]).type(torch.FloatTensor),
            torch.tensor(self.neighbours[idx]),
            torch.tensor(self.neighbour_cords[idx]).type(torch.FloatTensor),
            torch.tensor(self.mask[idx]).type(torch.FloatTensor),
            torch.tensor(self.labels[idx]).type(torch.FloatTensor)
        )
