import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, hidden_dim, input_dim=19, seq_len=10, num_layers=1, batch_size=10, device=torch.device('cuda')):
        super(LSTM, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.hidden_cell = (torch.zeros(2, seq_len, hidden_dim),
                            torch.zeros(2, seq_len, hidden_dim))
        self.dropout = nn.Dropout(p=0.1)
        self.activation = nn.ReLU()  # nn.Tanh()
        self.linear = nn.Linear(hidden_dim, 6)

    def forward(self, input_seq):
        input_seq = input_seq.to(self.device)
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(input_seq.size(1), input_seq.size(0), -1),
                                               self.hidden_cell)
        out = lstm_out.permute(1, 0, 2)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear(out)
        out[:, :, :3] *= 10000
        return out

    def init_hidden_cell(self):
        self.hidden_cell = (torch.zeros(self.num_layers, self.seq_len, self.hidden_dim).to(self.device),
                            torch.zeros(self.num_layers, self.seq_len, self.hidden_dim).to(self.device))
