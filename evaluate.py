import torch
from torch.utils import data
from utils import config
# from network.model import Model
from network import dataset
from sklearn.metrics import recall_score
from focal_loss.focal_loss import FocalLoss


def evaluate(model, val_dataloader, criterion):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    val_accuracy = 0.0
    val_loss = 0.0
    y_preds = []
    y_labels = []

    with torch.no_grad():
        for val_field, val_candidate, val_words, val_positions, masks, val_labels in val_dataloader:
            val_field = val_field.to(device)
            val_candidate = val_candidate.to(device)
            val_words = val_words.to(device)
            val_positions = val_positions.to(device)
            masks = masks.to(device)
            val_labels = val_labels.to(device)

            val_outputs = model(val_field, val_candidate, val_words, val_positions, masks)
            validation_loss = criterion(val_outputs, val_labels)

            val_preds = val_outputs.round()
            y_preds.extend(list(val_preds.cpu().detach().numpy().reshape(1, -1)[0]))
            y_labels.extend(list(val_labels.cpu().detach().numpy().reshape(1, -1)[0]))

            val_accuracy += torch.sum(val_preds == val_labels).item()
            val_loss += validation_loss.item()

        val_loss = val_loss / val_dataloader.sampler.num_samples
        val_accuracy = val_accuracy / val_dataloader.sampler.num_samples
        recall = recall_score(y_labels, y_preds)

    return val_accuracy, val_loss, recall


if __name__ == '__main__':

    doc_data = dataset.DocumentsDataset(split_name='val')
    VOCAB_SIZE = len(doc_data.vocab)
    test_data = data.DataLoader(doc_data, batch_size=config.BATCH_SIZE, shuffle=True)

    # rlie = Model(VOCAB_SIZE, config.EMBEDDING_SIZE, config.NEIGHBOURS, config.HEADS)
    # criterion = nn.BCELoss()
    relie = torch.load('output/model.pth')
    criterion = FocalLoss(alpha=2, gamma=5)

    test_accuracy, test_loss, test_recall = evaluate(relie, test_data, criterion)
    print(f"Test Accuracy: {test_accuracy} Test Loss: {test_loss} Test Recall: {test_recall}")
