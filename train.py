import torch
import torch.nn as nn
from torch.utils import data
import utils.constants as constants
from network.model import Model
from network import dataset
from evaluate import evaluate
from sklearn.metrics import recall_score
from focal_loss.focal_loss import FocalLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(model, train_dataloader, val_dataloader, epochs):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    writer = SummaryWriter(comment=f"LR_{constants.LR}_BATCH_{constants.BATCH_SIZE}")
    # criterion = nn.BCELoss()
    criterion = FocalLoss(alpha=2, gamma=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=constants.LR)

    train_loss_history = []
    train_accuracy_history = []
    recall_history = []
    val_loss_history = []
    val_accuracy_history = []
    val_recall_history = []

    for epoch in range(1, epochs + 1):

        train_loss = 0.0
        train_accuracy = 0.0
        y_preds = []
        y_labels = []

        for field, candidate, words, positions, masks, labels in tqdm(train_dataloader, desc="Epoch %s" % epoch):

            field = field.to(device)
            candidate = candidate.to(device)
            words = words.to(device)
            positions = positions.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            outputs = model(field, candidate, words, positions, masks)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = outputs.round()
            y_preds.extend(list(preds.cpu().detach().numpy().reshape(1, -1)[0]))
            y_labels.extend(list(labels.cpu().detach().numpy().reshape(1, -1)[0]))

            train_accuracy += torch.sum(preds == labels).item()
            train_loss += loss.item()

        else:

            val_accuracy, val_loss, val_recall = evaluate(model, val_dataloader, criterion)

            train_loss = train_loss / train_dataloader.sampler.num_samples
            train_accuracy = train_accuracy / train_dataloader.sampler.num_samples
            recall = recall_score(y_labels, y_preds)
            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_accuracy)
            recall_history.append(recall)

            val_loss_history.append(val_loss)
            val_accuracy_history.append(val_accuracy)
            val_recall_history.append(val_recall)

            writer.add_scalar('Recall/train', recall, epoch)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Recall/validation', val_recall, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

            print(f"Metrics for Epoch - {epoch}  Loss:{round(train_loss, 4)} \
                    Recall: {round(recall, 4)} \
                    Validation Loss: {round(val_loss, 4)} \
                    Validation Recall: {round(val_recall, 4)}")

    writer.flush()
    writer.close()
    return {
        'training_loss': train_loss_history,
        'training_accuracy': train_accuracy_history,
        'training_recall': recall_history,
        'validation_loss': val_loss_history,
        'validation_accuracy': val_accuracy_history,
        'validation_recall': recall_history
    }


if __name__ == '__main__':

    doc_data = dataset.DocumentsDataset(constants.XMLS, constants.OCR,
                                        constants.CANDIDATES, constants.OUTPUT_PATH,
                                        constants.NEIGHBOURS, constants.VOCAB_SIZE)
    VOCAB_SIZE = len(doc_data.vocab)
    VAL_DATA_LEN = int(len(doc_data) * constants.VAL_SPLIT)
    TRAIN_DATA_LEN = len(doc_data) - VAL_DATA_LEN
    train_set, val_set = data.random_split(doc_data, [TRAIN_DATA_LEN, VAL_DATA_LEN])

    training_data = data.DataLoader(train_set, batch_size=constants.BATCH_SIZE, shuffle=True)
    validation_data = data.DataLoader(val_set, batch_size=constants.BATCH_SIZE, shuffle=True)

    rlie = Model(VOCAB_SIZE, constants.EMBEDDING_SIZE, constants.NEIGHBOURS, constants.HEADS)

    history = train(rlie, training_data, validation_data, constants.EPOCHS)
    print(history)
