import torch
import torch.nn as nn
from torch.utils import data
import utils.constants as constants
from network.model import Model
from network import dataset


def evaluate(model, val_dataloader, criterion):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    val_accuracy = 0.0
    val_loss = 0.0

    with torch.no_grad():
        for val_field, val_candidate, val_words, val_positions, val_labels in val_dataloader:
            val_field = val_field.to(device)
            val_candidate = val_candidate.to(device)
            val_words = val_words.to(device)
            val_positions = val_positions.to(device)
            val_labels = val_labels.to(device)

            val_outputs = model(val_field, val_candidate, val_words, val_positions)
            validation_loss = criterion(val_outputs, val_labels)

            val_preds = val_outputs.round()
            val_accuracy += torch.sum(val_preds == val_labels).item()
            val_loss += validation_loss.item()

        val_loss = val_loss / val_dataloader.sampler.num_samples
        val_accuracy = val_accuracy / val_dataloader.sampler.num_samples

    return val_accuracy, val_loss


if __name__ == '__main__':

    doc_data = dataset.DocumentsDataset(constants.XMLS, constants.OCR,
                                        constants.IMAGES, constants.CANDIDATES,
                                        constants.FIELDS, constants.NEIGHBOURS)
    VOCAB_SIZE = len(doc_data.vocab)
    test_data = data.DataLoader(doc_data, batch_size=constants.BATCH_SIZE, shuffle=True)

    rlie = Model(VOCAB_SIZE, constants.EMBEDDING_SIZE, constants.NEIGHBOURS, constants.HEADS)
    criterion = nn.BCELoss()

    test_accuracy, test_loss = evaluate(rlie, test_data, criterion)
    print(f"Test Accuracy: {test_accuracy} Test Loss: {test_loss}")
