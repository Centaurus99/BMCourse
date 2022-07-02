import math
import torch
import torch.nn as nn
import torch.optim as optim
import time

import data
import model

torch.manual_seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = 'data/glue-sst2'
log_interval = 100
batch_size = 4
epochs = 10
save = 'model.pt'
lr = 0.01

dev_dataloader, test_dataloader, train_dataloader, dictionary = data.get_loader(
    path, batch_size)

print('=' * 89)
try:
    with open(save, 'rb') as f:
        checkpoint = torch.load(f)
        model = checkpoint['model']
        model.rnn.flatten_parameters()
        optimizer = checkpoint['optimizer']
        criterion = checkpoint['criterion']
        print('| Model loaded from {}'.format(save))

except FileNotFoundError:
    print('| Model not found, creating new model')
    model = model.RNN(len(dictionary), 100, 2, device=device)
    optimizer = optim.Adadelta(
        model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    print('| Model created')
print('=' * 89)


def accuracy(data_source):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for index, (texts, labels, _) in enumerate(data_source):
            texts = texts.to(device)
            labels = labels.to(device)
            output, hidden = model(texts, model.init_hidden(len(texts[0])))
            _, predicted = torch.max(output[-1], 1)
            _, labels = torch.max(labels, 1)
            total_correct += (predicted == labels).sum().item()
    return total_correct / data_source.dataset.__len__()


def evaluate(data_source):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for index, (texts, labels, _) in enumerate(data_source):
            texts = texts.to(device)
            labels = labels.to(device)
            output, hidden = model(texts, model.init_hidden(len(texts[0])))
            loss = criterion(output[-1], labels)
            total_loss += loss.item()
    return total_loss / len(data_source)


def train(data_source):
    model.train()
    total_loss = 0
    start_time = time.time()
    for index, (texts, labels, _) in enumerate(data_source):
        texts = texts.to(device)
        labels = labels.to(device)
        model.zero_grad()
        optimizer.zero_grad()
        output, hidden = model(texts, model.init_hidden(len(texts[0])))
        loss = criterion(output[-1], labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (index % log_interval == 0 and index > 0) or index == len(data_source) - 1:
            elapsed = time.time() - start_time
            if (index % log_interval == 0):
                cur_loss = total_loss / log_interval
                elapsed = elapsed * 1000 / log_interval
            else:
                cur_loss = total_loss / (index % log_interval)
                elapsed = elapsed * 1000 / (index % log_interval)
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.3f} | ms/batch {:5.2f} | '
                  'loss {:5.4f} | ppl {:8.4f}'.format(
                      epoch, index + 1, len(data_source),
                      optimizer.param_groups[0]['lr'],
                      elapsed, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


try:
    best_val_loss = None
    for epoch in range(1, epochs):
        epoch_start = time.time()
        train(train_dataloader)
        val_loss = evaluate(dev_dataloader)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '
              'valid ppl {:8.4f}'.format(epoch, (time.time() - epoch_start),
                                         val_loss, math.exp(val_loss)))
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model': model, 'optimizer': optimizer,
                        'criterion': criterion}, save)
            print('| Model saved to {}'.format(save))
        print('-' * 89)

except KeyboardInterrupt:
    print('-' * 89)
    print('| Exiting from training early')
    torch.save({
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion
    }, save + '_interrupt')
    print('| Model saved to {}'.format(save + '_interrupt'))
    print('-' * 89)

with open(save, 'rb') as f:
    model = torch.load(f)['model']
    model.rnn.flatten_parameters()

test_loss = evaluate(test_dataloader)
test_acc = accuracy(test_dataloader)
print('=' * 89)
print('| End of training | test loss {:5.4f} | test ppl {:8.4f} | test accuracy {:8.2%}'.format(
    test_loss, math.exp(test_loss), test_acc))
print('=' * 89)
