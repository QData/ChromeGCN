import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stop
import math
from torch.nn.parameter import Parameter

class Expecto(nn.Module):
    """An implementation of the ExPecto model described in "Deep learning 
    sequence-based ab initio prediction of variant effects on expression and 
    disease risk", Zhou et al., 2018.

    https://www.nature.com/articles/s41588-018-0160-6

    Args:
        nclass (int): The number of classes to predict
        seq_length (int): The length of the input sequence window

    """

    def __init__(self, nclass, seq_length):
        super(Expecto, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4
        sequence_length = seq_length
        n_targets=nclass
        linear_size = 128
        self.src_word_emb = nn.Embedding(5, 5)

        self.conv_net = nn.Sequential(
            nn.Conv1d(5, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(320, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(320),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.5))

        reduce_by = 2 * (conv_kernel_size - 1)
        pool_kernel_size = float(pool_kernel_size)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        
        self.linear = nn.Linear(960 * self._n_channels, linear_size)
        self.batch_norm = nn.BatchNorm1d(linear_size)
        self.classifier = nn.Linear(linear_size, n_targets)
        self.relu = nn.ReLU()


    def forward(self, x):
        """Predicts the epigenomic state of a given DNA sequence window

        Args:
            x (tensor): The DNA sequence window

        Returns:
            The features and the predicted epigenomic state for the input DNA sequence window

        """

        x = self.src_word_emb(x)
        out = self.conv_net(x.permute(0,2,1))
        reshape_out = out.view(out.size(0), 960 * self._n_channels)
        x_feat = self.linear(reshape_out)
        predict = self.relu(x_feat)
        predict = self.batch_norm(predict)
        predict = self.classifier(predict)
        return x_feat,predict,None

class DeepSEA(nn.Module):
    """An implementation of the DeepSEA model described in "Predicting effects
    of noncoding variants with deep learning–based sequence model", Zhou and Troyanskaya,
    2015.

    https://www.nature.com/articles/nmeth.3547 

     Args:
        nclass (int): The number of classes to predict
        seq_length (int): The length of the input sequence window

    """
    def __init__(self, nclass, seq_length):
        super(DeepSEA, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4
        sequence_length = seq_length
        n_targets=nclass
        linear_size = 128
        self.src_word_emb = nn.Embedding(5, 5)
        self.conv_net = nn.Sequential(
            nn.Conv1d(5, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),
            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        reduce_by = conv_kernel_size - 1
        pool_kernel_size = float(pool_kernel_size)
        self.n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        
        self.linear = nn.Linear(960 * self.n_channels, linear_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(linear_size, n_targets)


    def forward(self, x):
        """Predicts the epigenomic state of a given DNA sequence window

        Args:
            x (tensor): The DNA sequence window

        Returns:
            The features and the predicted epigenomic state for the input DNA sequence window

        """
        x = self.src_word_emb(x)
        out = self.conv_net(x.permute(0,2,1))
        reshape_out = out.view(out.size(0), 960 * self.n_channels)
        x_feat = self.linear(reshape_out)
        predict = self.relu(x_feat)
        if hasattr(self, 'dropout'):
            predict = self.dropout(predict)
        predict = self.classifier(x_feat)
        return x_feat,predict,None

class DanQ(nn.Module):
    """An implementation of the DanQ model described in "DanQ: a hybrid 
    convolutional and recurrent neural network for predicting the function 
    of DNA sequences", Quang and Xie, 2015. 

    https://www.biorxiv.org/content/10.1101/032821v1 

    Args:
        nclass (int): The number of classes to predict

    """

    def __init__(self, nclass):
        super(DanQ, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=5, out_channels=320, kernel_size=26)
        self.Maxpool = nn.MaxPool1d(kernel_size=13, stride=13)
        self.Drop1 = nn.Dropout(p=0.2)
        self.BiLSTM = nn.LSTM(input_size=320, hidden_size=320, num_layers=2,
                                 batch_first=True,
                                 dropout=0.5,
                                 bidirectional=True)
        self.Linear1 = nn.Linear(640*151, 925)
        self.Linear2 = nn.Linear(925, nclass)
        self.src_word_emb = nn.Embedding(5, 5)

    def forward(self, input):
        """Predicts the epigenomic state of a given DNA sequence window

        Args:
            input (tensor): The DNA sequence window

        Returns:
            The features and the predicted epigenomic state for the input DNA sequence window

        """
        x = self.src_word_emb(input).permute(0,2,1)
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x_x = torch.transpose(x, 1, 2)
        x, (h_n,h_c) = self.BiLSTM(x_x)
        x = x.contiguous().view(-1, 640*151)
        x = self.Linear1(x)
        x_feat = F.relu(x)
        x = self.Linear2(F.relu(x))
        return x_feat,x,None






