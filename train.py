import torch
import torch.nn as nn
from torch.utils import data
import utils.constants as constants
from network.model import Model
from network import dataset


def train(model, train_dataloader, val_dataloader, epochs):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    for epoch in range(1, epochs + 1):

        train_loss = 0.0
        train_accuracy = 0.0
        val_loss = 0.0
        val_accuracy = 0.0

        for field, candidate, words, positions, labels in train_dataloader:

            field = field.to(device)
            candidate = candidate.to(device)
            words = words.to(device)
            positions = positions.to(device)
            labels = labels.to(device)

            outputs = model(field, candidate, words, positions)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)

            train_accuracy += torch.sum(preds == labels).item()
            train_loss += loss.item()

        else:

            with torch.no_grad():

                for val_field, val_candidate, val_words, val_positions, val_labels in val_dataloader:
                    val_field = val_field.to(device)
                    val_candidate = val_candidate.to(device)
                    val_words = val_words.to(device)
                    val_positions = val_positions.to(device)
                    val_labels = val_labels.to(device)

                    val_outputs = model(val_field, val_candidate, val_words, val_positions)
                    validation_loss = criterion(val_outputs, val_labels)

                    _, val_preds = torch.max(val_outputs, 1)
                    val_accuracy += torch.sum(val_preds == val_labels).item()
                    val_loss += validation_loss.item()

            train_loss = train_loss / len(train_dataloader)
            train_accuracy = train_accuracy / len(train_dataloader)
            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_accuracy)

            val_loss = val_loss / len(val_dataloader)
            val_accuracy = val_accuracy / len(val_dataloader)
            val_loss_history.append(val_loss)
            val_accuracy_history.append(val_accuracy)

            print(f"Epoch:{epoch} Loss:{round(train_loss, 4)} \
                  Accuracy: {round(train_accuracy, 4)} \
                  Validation Loss: {round(val_loss, 4)} \
                  Validation Accuracy: {round(val_accuracy, 4)}")

    training_history = {
        'training_loss': train_loss_history,
        'training_accuracy': train_accuracy_history,
        'validation_loss': val_loss_history,
        'validation_accuracy': val_accuracy_history
    }
    return training_history


if __name__ == '__main__':

    doc_data = dataset.DocumentsDataset(constants.XMLS, constants.OCR,
                                        constants.IMAGES, constants.CANDIDATES,
                                        constants.FIELDS, constants.NEIGHBOURS)
    VOCAB_SIZE = len(doc_data.vocab)
    VAL_DATA_LEN = int(len(doc_data) * constants.VAL_SPLIT)
    TRAIN_DATA_LEN = len(doc_data) - VAL_DATA_LEN
    train_set, val_set = data.random_split(doc_data, [TRAIN_DATA_LEN, VAL_DATA_LEN])

    training_data = data.DataLoader(train_set, batch_size=constants.BATCH_SIZE, shuffle=True)
    validation_data = data.DataLoader(val_set, batch_size=constants.BATCH_SIZE, shuffle=True)

    rlie = Model(VOCAB_SIZE, constants.EMBEDDING_SIZE, constants.NEIGHBOURS, constants.HEADS)

    history = train(rlie, training_data, validation_data, constants.EPOCHS)
