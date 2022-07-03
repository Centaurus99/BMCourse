import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, output_size, nlayers, dropout, device):
        super(RNN, self).__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.device = device
        self.embedding = nn.Embedding(input_size, hidden_size, device=device)
        self.rnn = getattr(nn, rnn_type)(
            hidden_size, hidden_size, nlayers, dropout=dropout if nlayers > 1 else 0, device=device)
        self.fc = nn.Linear(hidden_size, output_size, device=device)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, x, hidden):
        x = self.embedding(x)
        x, hidden = self.rnn(x, hidden)
        x = self.fc(x)
        x = self.dropout(x)
        return x, hidden

    def init_hidden(self, batch_size):
        if self.rnn_type == 'LSTM':
            return (torch.zeros(self.nlayers, batch_size, self.hidden_size, device=self.device),
                    torch.zeros(self.nlayers, batch_size, self.hidden_size, device=self.device))
        else:
            return torch.zeros(self.nlayers, batch_size, self.hidden_size, device=self.device)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.zero_()
