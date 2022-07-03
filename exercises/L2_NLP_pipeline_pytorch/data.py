import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, word):
        return self.word2idx[word]


class SST2(Dataset):
    def __init__(self, path, dictionary):
        self.dictionary = dictionary
        self.data = self.tokenize(path)

    def tokenize(self, path):
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            for line in f:
                text, label = line.split('\t')
                words = text.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            datas = []
            for line in f:
                text, label = line.split('\t')
                words = text.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                datas.append((torch.tensor(ids), label))
        return datas

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = [len(sq[0]) for sq in data]
    x = [i[0] for i in data]
    y = []
    for i in data:
        assert i[1].strip() == 'positive' or i[1].strip() == 'negative'
        if i[1].strip() == 'positive':
            y.append(torch.tensor([0.0, 1.0]))
        else:
            y.append(torch.tensor([1.0, 0.0]))
    y = torch.stack(y)
    data = pad_sequence(x, padding_value=0)
    return data, y, data_length


def get_loader(path, batch_size):
    dictionary = Dictionary()
    dev_dataset = SST2(os.path.join(path, 'dev.txt'), dictionary)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size,
                                shuffle=True, collate_fn=collate_fn)
    test_dataset = SST2(os.path.join(path, 'test.txt'), dictionary)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=True, collate_fn=collate_fn)
    train_dataset = SST2(os.path.join(path, 'train.txt'), dictionary)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_fn)
    return dev_dataloader, test_dataloader, train_dataloader, dictionary
