import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, hidden_dim, input_dim=19, seq_len=10, num_layers=1, batch_size=10):
        super(LSTM, self).__init__()
        # self.batchnorm = nn.BatchNorm1d(13)
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.hidden_cell = (torch.zeros(2, seq_len, hidden_dim),
                            torch.zeros(2, seq_len, hidden_dim))
        self.dropout = nn.Dropout(p=0.1)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(hidden_dim, 6)

    def forward(self, input_seq):
        # input_seq = input_seq.permute(0, 2, 1)
        # input_seq = self.batchnorm(input_seq)
        # input_seq = input_seq.permute(0, 2, 1)
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(input_seq.size(1), input_seq.size(0), -1),
                                               self.hidden_cell)
        out = lstm_out.permute(1, 0, 2)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear(out)
        out[:, :, :3] *= 10000
        return out

    def init_hidden_cell(self):
        self.hidden_cell = (torch.zeros(self.num_layers, self.seq_len, self.hidden_dim),
                            torch.zeros(self.num_layers, self.seq_len, self.hidden_dim))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 13

        self.conv1 = nn.Conv1d(13, 13, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(13)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(256, 6)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])
