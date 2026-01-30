"""
Baseline Models for Comparison
包括：LSTM, RNN, TCN, FC-ANN, Dropout-ANN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    """
    LSTM 基线模型
    """
    def __init__(self, input_dim=8, hidden_dim=64, num_classes=5, 
                 num_layers=5, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, 
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0,
                           bidirectional=False)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x: [batch, seq, features]
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 使用最后时刻的隐藏状态
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden]
        return self.fc(last_hidden)


class RNNClassifier(nn.Module):
    """
    RNN 基线模型
    """
    def __init__(self, input_dim=8, hidden_dim=64, num_classes=5, 
                 num_layers=5, dropout=0.3):
        super(RNNClassifier, self).__init__()
        
        self.rnn = nn.RNN(input_dim, hidden_dim,
                         num_layers=num_layers,
                         batch_first=True,
                         dropout=dropout if num_layers > 1 else 0)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        rnn_out, h_n = self.rnn(x)
        last_hidden = rnn_out[:, -1, :]
        return self.fc(last_hidden)


class TCNClassifier(nn.Module):
    """
    Temporal Convolutional Network (TCN) 基线模型
    使用因果卷积和残差连接
    """
    def __init__(self, input_dim=8, num_channels=[64, 64, 64], 
                 kernel_size=3, num_classes=5, dropout=0.3):
        super(TCNClassifier, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # 因果卷积
            padding = (kernel_size - 1) * (2 ** i)  # 膨胀因果卷积
            
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         padding=padding, dilation=2**i),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
    def forward(self, x):
        # x: [batch, seq, features] -> [batch, features, seq]
        x = x.transpose(1, 2)
        out = self.network(x)  # [batch, out_ch, seq]
        
        # 全局平均池化
        out = out.mean(dim=2)  # [batch, out_ch]
        return self.fc(out)


class ANNClassifier(nn.Module):
    """
    全连接人工神经网络 (FC-ANN)
    """
    def __init__(self, input_dim=8, seq_len=10, hidden_dims=[256, 128, 64], 
                 num_classes=5, dropout=0.0):
        super(ANNClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim * seq_len
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(prev_dim, num_classes)
        self.seq_len = seq_len
        
    def forward(self, x):
        # 展平时序特征
        batch = x.size(0)
        x = x.view(batch, -1)  # [batch, seq*features]
        features = self.network(x)
        return self.fc(features)


class DropoutANN(ANNClassifier):
    """
    带Dropout的ANN (对应论文4.6节)
    """
    def __init__(self, input_dim=8, seq_len=10, hidden_dims=[256, 128, 64], 
                 num_classes=5, dropout=0.3):
        super().__init__(input_dim, seq_len, hidden_dims, num_classes, dropout)