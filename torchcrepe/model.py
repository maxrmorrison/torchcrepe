import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class Crepe(nn.Module):

    
    def __init__(self, weight_file):
        super().__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.conv1 = self.__conv(2, name='conv1', in_channels=1, out_channels=128, kernel_size=(512, 1), stride=(4, 1), groups=1, bias=True)
        self.conv1_BN = self.__batch_normalization(2, 'conv1-BN', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2 = self.__conv(2, name='conv2', in_channels=128, out_channels=16, kernel_size=(64, 1), stride=(1, 1), groups=1, bias=True)
        self.conv2_BN = self.__batch_normalization(2, 'conv2-BN', num_features=16, eps=0.0010000000474974513, momentum=0.0)
        self.conv3 = self.__conv(2, name='conv3', in_channels=16, out_channels=16, kernel_size=(64, 1), stride=(1, 1), groups=1, bias=True)
        self.conv3_BN = self.__batch_normalization(2, 'conv3-BN', num_features=16, eps=0.0010000000474974513, momentum=0.0)
        self.conv4 = self.__conv(2, name='conv4', in_channels=16, out_channels=16, kernel_size=(64, 1), stride=(1, 1), groups=1, bias=True)
        self.conv4_BN = self.__batch_normalization(2, 'conv4-BN', num_features=16, eps=0.0010000000474974513, momentum=0.0)
        self.conv5 = self.__conv(2, name='conv5', in_channels=16, out_channels=32, kernel_size=(64, 1), stride=(1, 1), groups=1, bias=True)
        self.conv5_BN = self.__batch_normalization(2, 'conv5-BN', num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.conv6 = self.__conv(2, name='conv6', in_channels=32, out_channels=64, kernel_size=(64, 1), stride=(1, 1), groups=1, bias=True)
        self.conv6_BN = self.__batch_normalization(2, 'conv6-BN', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.classifier = self.__dense(name = 'classifier', in_features = 256, out_features = 360, bias = True)

    def forward(self, x):
        input_reshape   = x[:, None, :, None]
        conv1_pad       = F.pad(input_reshape, (0, 0, 254, 254))
        conv1           = self.conv1(conv1_pad)
        conv1_activation = F.relu(conv1)
        conv1_BN        = self.conv1_BN(conv1_activation)
        conv1_maxpool, conv1_maxpool_idx = F.max_pool2d(conv1_BN, kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=False, return_indices=True)
        conv1_dropout   = F.dropout(input = conv1_maxpool, p = 0.25, training = self.training, inplace = True)
        conv2_pad       = F.pad(conv1_dropout, (0, 0, 31, 32))
        conv2           = self.conv2(conv2_pad)
        conv2_activation = F.relu(conv2)
        conv2_BN        = self.conv2_BN(conv2_activation)
        conv2_maxpool, conv2_maxpool_idx = F.max_pool2d(conv2_BN, kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=False, return_indices=True)
        conv2_dropout   = F.dropout(input = conv2_maxpool, p = 0.25, training = self.training, inplace = True)
        conv3_pad       = F.pad(conv2_dropout, (0, 0, 31, 32))
        conv3           = self.conv3(conv3_pad)
        conv3_activation = F.relu(conv3)
        conv3_BN        = self.conv3_BN(conv3_activation)
        conv3_maxpool, conv3_maxpool_idx = F.max_pool2d(conv3_BN, kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=False, return_indices=True)
        conv3_dropout   = F.dropout(input = conv3_maxpool, p = 0.25, training = self.training, inplace = True)
        conv4_pad       = F.pad(conv3_dropout, (0, 0, 31, 32))
        conv4           = self.conv4(conv4_pad)
        conv4_activation = F.relu(conv4)
        conv4_BN        = self.conv4_BN(conv4_activation)
        conv4_maxpool, conv4_maxpool_idx = F.max_pool2d(conv4_BN, kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=False, return_indices=True)
        conv4_dropout   = F.dropout(input = conv4_maxpool, p = 0.25, training = self.training, inplace = True)
        conv5_pad       = F.pad(conv4_dropout, (0, 0, 31, 32))
        conv5           = self.conv5(conv5_pad)
        conv5_activation = F.relu(conv5)
        conv5_BN        = self.conv5_BN(conv5_activation)
        conv5_maxpool, conv5_maxpool_idx = F.max_pool2d(conv5_BN, kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=False, return_indices=True)
        conv5_dropout   = F.dropout(input = conv5_maxpool, p = 0.25, training = self.training, inplace = True)
        conv6_pad       = F.pad(conv5_dropout, (0, 0, 31, 32))
        conv6           = self.conv6(conv6_pad)
        conv6_activation = F.relu(conv6)
        conv6_BN        = self.conv6_BN(conv6_activation)
        conv6_maxpool, conv6_maxpool_idx = F.max_pool2d(conv6_BN, kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=False, return_indices=True)
        conv6_dropout   = F.dropout(input = conv6_maxpool, p = 0.25, training = self.training, inplace = True)
        flatten         = conv6_dropout.reshape(-1, 256)
        classifier      = self.classifier(flatten)
        classifier_activation = F.sigmoid(classifier)
        return classifier_activation


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in __weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(__weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(__weights_dict[name]['var']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

