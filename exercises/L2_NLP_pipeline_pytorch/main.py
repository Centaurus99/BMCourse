import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import time

import data
import model

parser = argparse.ArgumentParser(description='PyTorch NLP Pipeline')
parser.add_argument('--path', type=str, default='data/glue-sst2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='RNN',
                    help='type of recurrent net (LSTM, GRU, RNN)')
parser.add_argument('--nhid', type=int, default=1000,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--lr', type=float, default=0.05,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
parser.add_argument('--log_interval', type=int, default=500,
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dev_dataloader, test_dataloader, train_dataloader, dictionary = data.get_loader(
    args.path, args.batch_size)

print('=' * 89)
try:
    with open(args.save, 'rb') as f:
        checkpoint = torch.load(f)
        model = checkpoint['model']
        model.rnn.flatten_parameters()
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
        criterion = checkpoint['criterion']
        print('| Model loaded from {}'.format(args.save))

except FileNotFoundError:
    print('| Model not found, creating new model')
    model = model.RNN(args.model, len(dictionary), args.nhid, 2, args.nlayers,
                      args.dropout, device=device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=4, verbose=True)
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
        if (index % args.log_interval == 0 and index > 0) or index == len(data_source) - 1:
            elapsed = time.time() - start_time
            if (index % args.log_interval == 0):
                cur_loss = total_loss / args.log_interval
                elapsed = elapsed * 1000 / args.log_interval
            else:
                cur_loss = total_loss / (index % args.log_interval)
                elapsed = elapsed * 1000 / (index % args.log_interval)
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.4f} | ms/batch {:5.2f} | '
                  'loss {:5.4f} | ppl {:8.4f}'.format(
                      epoch, index + 1, len(data_source),
                      optimizer.param_groups[0]['lr'],
                      elapsed, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


try:
    best_val_loss = None
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train(train_dataloader)
        val_loss = evaluate(dev_dataloader)
        scheduler.step(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '
              'valid ppl {:8.4f}'.format(epoch, (time.time() - epoch_start),
                                         val_loss, math.exp(val_loss)))
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model': model, 'optimizer': optimizer, 'scheduler': scheduler,
                        'criterion': criterion}, args.save)
            print('| Model saved to {}'.format(args.save))
        print('-' * 89)

except KeyboardInterrupt:
    print('-' * 89)
    print('| Exiting from training early')
    torch.save({
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion
    }, args.save + '_interrupt')
    print('| Model saved to {}'.format(args.save + '_interrupt'))
    print('-' * 89)

with open(args.save, 'rb') as f:
    model = torch.load(f)['model']
    model.rnn.flatten_parameters()

test_loss = evaluate(test_dataloader)
test_acc = accuracy(test_dataloader)
print('=' * 89)
print('| End of training | test loss {:5.4f} | test ppl {:8.4f} | test accuracy {:8.2%}'.format(
    test_loss, math.exp(test_loss), test_acc))
print('=' * 89)
