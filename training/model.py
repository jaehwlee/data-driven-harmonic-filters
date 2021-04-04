# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modules import HarmonicSTFT

class Model(nn.Module):
    def __init__(self,
                conv_channels=128,
                sample_rate=16000,
                n_fft=513,
                n_harmonic=6,
                semitone_scale=2,
                learn_bw=None,
                dataset='mtat'):
        super(Model, self).__init__()

        self.hstft = HarmonicSTFT(sample_rate=sample_rate,
                                    n_fft=n_fft,
                                    n_harmonic=n_harmonic,
                                    semitone_scale=semitone_scale,
                                    learn_bw=learn_bw)
        self.hstft_bn = nn.BatchNorm2d(n_harmonic)

        # 2D CNN
        if dataset == 'mtat':
            from modules import ResNet_mtat as ResNet
        elif dataset == 'dcase':
            from modules import ResNet_dcase as ResNet
        elif dataset == 'keyword':
            from modules import ResNet_keyword as ResNet
        self.conv_2d = ResNet(input_channels=n_harmonic, conv_channels=conv_channels)

    def forward(self, x):
        # harmonic stft
        x = self.hstft_bn(self.hstft(x))

        # 2D CNN
        logits = self.conv_2d(x)

        return logits




########################################
# for ae
###########################################

class WaveEncoder(nn.Module):
    def __init__(self,
                conv_channels=128,
                sample_rate=16000,
                n_fft=513,
                n_harmonic=6,
                semitone_scale=2,
                learn_bw=None,
                dataset='mtat'):
        super(WaveEncoder, self).__init__()

        self.hstft = HarmonicSTFT(sample_rate=sample_rate,
                                    n_fft=n_fft,
                                    n_harmonic=n_harmonic,
                                    semitone_scale=semitone_scale,
                                    learn_bw=learn_bw)
        self.hstft_bn = nn.BatchNorm2d(n_harmonic)

        # 2D CNN
        if dataset == 'mtat':
            from modules import WaveEncoder_mtat as ResNet
        elif dataset == 'dcase':
            from modules import ResNet_dcase as ResNet
        elif dataset == 'keyword':
            from modules import ResNet_keyword as ResNet
        self.conv_2d = ResNet(input_channels=n_harmonic, conv_channels=conv_channels)

    def forward(self, x):
        # harmonic stft
        x = self.hstft_bn(self.hstft(x))

        # 2D CNN
        logits = self.conv_2d(x)

        return logits




class WaveProjector(nn.Module):
    """
    A projection network, P(·), which maps the normalized representation vector r into a vector z = P(r) ∈ R^{DP} 
    suitable for computation of the contrastive loss.
    """

    def __init__(self, input_dimension=256, project_dimension=128):
        super(WaveProjector, self).__init__()
        self.fc_1 = nn.Linear(input_dimension, input_dimension)
        self.bn = nn.BatchNorm1d(input_dimension)
        self.fc_2 = nn.Linear(input_dimension, project_dimension)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.LeakyReLU(0.3)

    def forward(self, x):
        # fully connected
        x = self.fc_1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        x = self.relu(x)
        return x





class SupervisedClassifier(nn.Module):
    """For stage 2, simply a softmax on top of the Encoder.
    """
    def __init__(self, input_dimension=256, project_dimension=50):
        super(SupervisedClassifier, self).__init__()
        self.fc_1 = nn.Linear(input_dimension, input_dimension)
        self.bn = nn.BatchNorm1d(input_dimension)
        self.bin2 = nn.BatchNorm1d(input_dimension)
        self.fc_2 = nn.Linear(input_dimension, input_dimenstion)
        self.dropout = nn.Dropout(0.5)
        self.fc_3 = nn.Linear(input_dimension, project_dimension)
        self.relu = nn.LeakyReLU(0.3)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # fully connected
        x = self.fc_1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.bin2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_3(x)
        x = self.activation(x)
        return x




class TagEncoder(nn.Module):
    """For stage 2, simply a softmax on top of the Encoder.
    """
    def __init__(self, input_dimension=50, project_dimension=128):
        super(TagEncoder, self).__init__()
        self.fc_1 = nn.Linear(input_dimension, project_dimension)
        self.bn = nn.BatchNorm1d(project_dimension)
        self.dropout = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(project_dimension, project_dimension)
        self.relu = nn.LeakyReLU(0.3)

    def forward(self, x):
        # fully connected
        x = self.fc_1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x






class TagDecoder(nn.Module):
    """For stage 2, simply a softmax on top of the Encoder.
    """
    def __init__(self, output_dimension=50, project_dimension=128):
        super(TagDecoder, self).__init__()
        self.fc_1 = nn.Linear(project_dimension, project_dimension)
        self.relu = nn.LeakyReLU(0.3)
        self.bn = nn.BatchNorm1d(project_dimension)
        self.fc_2 = nn.Linear(project_dimension, output_dimension)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # fully connected
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.fc_2(x)
        x = self.activation(x)
        return x





class SupervisedClassifier(nn.Module):
    """For stage 2, simply a softmax on top of the Encoder.
    """
    def __init__(self, input_dimension=256, project_dimension=50):
        super(SupervisedClassifier, self).__init__()
        self.fc_1 = nn.Linear(input_dimension, input_dimension)
        self.bn = nn.BatchNorm1d(input_dimension)
        self.fc_2 = nn.Linear(input_dimension, project_dimension)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.LeakyReLU(0.3)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # fully connected
        x = self.fc_1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        x = self.activation(x)
        return x

