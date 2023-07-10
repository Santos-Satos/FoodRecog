import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class ResNet_Linear(nn.Module):
    
    def __init__(self, n_channel, n_output, drop_rate):
        super().__init__()
        
        self.Conv_1 = nn.Conv2d(1, n_channel, kernel_size=(128, 1), stride=(128, 1))
        self.ResNet_1 = ResNet(n_channel, n_channel*2, True, drop_rate)
        self.ResNet_2 = ResNet(n_channel*2, n_channel*4, True, drop_rate)
        self.ResNet_3 = ResNet(n_channel*4, n_channel*8, True, drop_rate)
        self.Pool = nn.MaxPool2d(kernel_size=(128, 13), stride=(128, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.Flatten = nn.Flatten(1, -1)
        self.fc1 = nn.Linear(1536, n_output)# bidirectionalだからhidden_size*2
        
        self.shape_check_flag = False

    def forward(self, x):
        if self.shape_check_flag:print(x.size())
        x = x.unsqueeze(dim=1)
        if self.shape_check_flag:print(x.size())
        x = self.Conv_1(x)
        if self.shape_check_flag:print(x.size())
        x = self.ResNet_1(x)
        if self.shape_check_flag:print(x.size())
        x = self.ResNet_2(x)
        if self.shape_check_flag:print(x.size())
        x = self.ResNet_3(x)
        if self.shape_check_flag:print(x.size())
        x = self.Pool(x)
        if self.shape_check_flag:print(x.size())
        x = self.Flatten(x)
        if self.shape_check_flag:print(x.size())
        x = self.dropout(x)
        if self.shape_check_flag:print(x.size())
        if self.shape_check_flag:exit()
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)

        return output

# ResNetのクラス
class ResNet(nn.Module):
    def __init__(self, in_channel_num, out_channel_num, exec_conv_flag, drop_rate):
        super().__init__()
        
        self.exec_conv_flag = exec_conv_flag

        self.BatchNorm_1 = nn.BatchNorm2d(in_channel_num)
        self.BatchNorm_2 = nn.BatchNorm2d(out_channel_num)
        self.ReLU = nn.ReLU()
        self.Conv_1 = nn.Conv2d(in_channel_num, out_channel_num, kernel_size=3, stride=1, padding=1)
        self.Conv_2 = nn.Conv2d(out_channel_num, out_channel_num, kernel_size=3, stride=1, padding=1)
        self.Dropout_1 = nn.Dropout(drop_rate)
        self.Dropout_2 = nn.Dropout(drop_rate)


        if self.exec_conv_flag == True:
            self.Conv_3 = nn.Conv2d(in_channel_num, out_channel_num, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        path_a = self.BatchNorm_1(x)
        path_a = self.ReLU(path_a)
        path_a = self.Dropout_1(path_a)
        path_a = self.Conv_1(path_a)
        path_a = self.BatchNorm_2(path_a)
        path_a = self.ReLU(path_a)
        path_a = self.Dropout_2(path_a)
        path_a = self.Conv_2(path_a)

        path_b = x
        if self.exec_conv_flag == True:
            path_b = self.Conv_3(x)

        return path_a + path_b
