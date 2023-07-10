import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class CNN_LSTM_Linear(nn.Module):
    
    def __init__(self, n_channel, n_output, drop_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(1, n_channel, kernel_size=(5, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.pool1 = nn.MaxPool2d((5, 2))
        
        self.conv2 = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=(5, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(2 * n_channel)
        self.pool2 = nn.MaxPool2d((5, 2))
        
        self.conv3 = nn.Conv2d(2 * n_channel, 4 * n_channel, kernel_size=(5, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(4 * n_channel)
        self.pool3 = nn.MaxPool2d((5, 2))
        
        self.lstm = nn.LSTM(input_size = 4*n_channel,
                    hidden_size = 256,
                    batch_first = True,
                    bidirectional=True)
        self.dropout = nn.Dropout(drop_rate)
        self.fc1 = nn.Linear(512, n_output)# bidirectionalだからhidden_size*2
        
        self.shape_check_flag = True

    def forward(self, x):
        if self.shape_check_flag:print(x.size())
        x = x.unsqueeze(dim=1)
        if self.shape_check_flag:print(x.size())
        x = self.conv1(x)
        if self.shape_check_flag:print(x.size())
        x = F.relu(self.bn1(x))
        if self.shape_check_flag:print(x.size())
        x = self.pool1(x)
        if self.shape_check_flag:print(x.size())
        x = self.conv2(x)
        if self.shape_check_flag:print(x.size())
        x = F.relu(self.bn2(x))
        if self.shape_check_flag:print(x.size())
        x = self.pool2(x)
        if self.shape_check_flag:print(x.size())
        x = self.conv3(x)
        if self.shape_check_flag:print(x.size())
        x = F.relu(self.bn3(x))
        if self.shape_check_flag:print(x.size())
        x = self.pool3(x)
        if self.shape_check_flag:print(x.size())
        x = x.permute(0, 2, 1, 3)
        if self.shape_check_flag:print(x.size())
        x = torch.squeeze(x, 3)
        if self.shape_check_flag:print(x.size())
        if self.shape_check_flag:exit()
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.dropout(x)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)

        return output

        
