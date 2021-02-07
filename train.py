import torch
from torch.utils import data
from utils import config
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
    writer = SummaryWriter(comment=f"LR_{config.LR}_BATCH_{config.BATCH_SIZE}")
    # criterion = nn.BCELoss()
    criterion = FocalLoss(alpha=2, gamma=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    train_loss_history = []
    train_accuracy_history = []
    recall_history = []
    val_loss_history = []
    val_accuracy_history = []
    val_recall_history = []
    val_max_score = 0.0
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
            if val_recall > val_max_score: # Saving the best model
                print('saving model....')
                val_max_score = val_recall
                torch.save(model, 'output/model.pth')
            print(f"Metrics for Epoch {epoch}:  Loss:{round(train_loss, 4)} \
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

    # split name must equal to split filename eg: for train.txt -> train
    train_data = dataset.DocumentsDataset(split_name='train')
    val_data = dataset.DocumentsDataset(split_name='val')

    VOCAB_SIZE = len(train_data.vocab)

    training_data = data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    validation_data = data.DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=True)

    relie = Model(VOCAB_SIZE, config.EMBEDDING_SIZE, config.NEIGHBOURS, config.HEADS)
    # relie = torch.load('output/model.pth')
    history = train(relie, training_data, validation_data, config.EPOCHS)
    print(history)
