import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BaselineNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.tanh1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(self.hidden_size, 6)
        self.tanh2 = torch.nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh2(x)
        return x


class LSTM(nn.Module):

    def __init__(self, hidden_dim, seq_len=2):
        super(LSTM, self).__init__()
        self.input_dim = 7
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(7, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(seq_len * self.hidden_dim, 6)
        self.tanh = nn.Tanh()

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out.reshape(batch_size, -1))
        predictions = self.tanh(predictions)
        return predictions