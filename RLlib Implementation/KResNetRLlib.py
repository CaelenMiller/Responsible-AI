import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import copy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from math import sqrt
from ray.rllib.utils.annotations import override


from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn



class ResNetLayer(nn.Module):

    def __init__(self, inputChannels1D, inputChannels2D, nChannels=1, activation="relu", batchNorm=False, dropoutRate=0):
        super(ResNetLayer, self).__init__()

        activationName = activation
        self.batchnorm = batchNorm
        self.nChannels = nChannels
        self.inputChannels1D = inputChannels1D
        self.inputChannels2D = inputChannels2D #must be a square

        self.dropout = nn.Dropout(dropoutRate)

        self.batchnormLayer1D = nn.BatchNorm1d(self.inputChannels1D)
        self.batchnormLayer2D = nn.BatchNorm2d(self.nChannels)

        self.linear1 = nn.Linear(inputChannels1D, inputChannels1D) # might need these to be a bit more descriptive  
        self.linear2 = nn.Linear(inputChannels1D, inputChannels1D)

        self.conv1 = nn.Conv2d(in_channels=self.nChannels, out_channels=self.nChannels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.nChannels, out_channels=self.nChannels, kernel_size=3, padding=1)

        if activationName == "relu":
            self.activation = nn.ReLU()
        elif activationName == "leakyrelu":
            self.activation = nn.LeakyReLU()
        elif activationName == "selu":
            self.activation = nn.SELU()
        elif activationName == "hardshrink":
            self.activation = nn.Hardshrink()
        elif activationName == "elu":
            self.activation = nn.ELU()
        else:
            print(activationName)

    def forward(self, x):
        spatial_state = x[:, :-self.inputChannels1D].view(-1, 1, int(sqrt(self.inputChannels2D)), int(sqrt(self.inputChannels2D))).float()
        non_spatial_state = x[:, -self.inputChannels1D:].float()

        #Deal with the spatial stuff first
        spatial_out = self.conv1(self.dropout(self.activation(self.conv1(self.dropout(spatial_state)))))
        non_spatial_out = self.linear1(self.dropout(self.activation(self.linear2(self.dropout(non_spatial_state)))))

        if self.batchnorm:
            spatial_out = self.batchnormLayer2D(spatial_out)
            non_spatial_out = self.batchnormLayer1D(non_spatial_out)
        
        spatial_out = spatial_state + spatial_out
        non_spatial_out = non_spatial_state + non_spatial_out

        # Flatten spatial data
        spatial_out_flattened = spatial_out.view(-1, self.inputChannels2D)

        # Combine back into the original format, ready to use for the next layer. 
        combined_out = torch.cat((spatial_out_flattened, non_spatial_out), dim=1)

        return combined_out


class KoopmanResNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)


        self.resChannels = kwargs["resChannels"]
        inChannels = kwargs["inChannels"]
        resChannels = kwargs["resChannels"]
        nLayers = kwargs["nLayers"]
        activation = kwargs["activation"]
        batchNorm = kwargs["batchNorm"]
        dropout = kwargs["dropout"]
        encodingDims = kwargs["encodingDims"]
        action_size = kwargs["actionSize"]
        initialization = kwargs["initialization"]
        self.inputChannels2D = kwargs["inputChannels2D"]
        self.inputChannels1D = kwargs["inputChannels1D"]
        self.inputSize = self.inputChannels2D + self.inputChannels1D


       # inChannels to resChannels (and vise-versa). Currently not in use, due to complications with 1D + 2D data
        self.inToRes = nn.Conv2d(in_channels=inChannels, out_channels=resChannels, kernel_size=3, padding=1)
        self.resToIn = nn.Conv2d(in_channels=resChannels, out_channels=inChannels, kernel_size=3, padding=1)
        # resLayers (and vis-versa)
        self.resLayersG = nn.ModuleList([])
        for layer in range(nLayers):
            self.resLayersG.append(ResNetLayer(inputChannels1D=self.inputChannels1D, inputChannels2D=self.inputChannels2D,
                                                nChannels=resChannels, activation=activation, batchNorm=batchNorm,dropoutRate=dropout))

        self.resLayersGInv = nn.ModuleList([])
        for layer in range(nLayers):
            self.resLayersGInv.append(ResNetLayer(inputChannels1D=self.inputChannels1D, inputChannels2D=self.inputChannels2D,
                                                nChannels=resChannels, activation=activation, batchNorm=batchNorm,dropoutRate=dropout))

        # outChannels to categories (and vise-versa)
        self.outToEncoding = nn.Linear(in_features=self.inputSize * resChannels, out_features=encodingDims)
        self.encodingToOut = nn.Linear(in_features=encodingDims, out_features=self.inputSize * resChannels)

        self.K = nn.Linear(encodingDims, action_size, bias=False)

        if initialization == "xh":
            nn.init.xavier_normal_(self.inToRes.weight)
            nn.init.xavier_normal_(self.resToIn.weight)
            nn.init.xavier_normal_(self.outToEncoding.weight)
            nn.init.xavier_normal_(self.encodingToOut.weight)

        elif initialization == "o":
            nn.init.orthogonal_(self.inToRes.weight)
            nn.init.orthogonal_(self.resToIn.weight)
            nn.init.orthogonal_(self.outToEncoding.weight)
            nn.init.orthogonal_(self.encodingToOut.weight)


        self.critic_head = nn.Linear(encodingDims, 1, bias=False)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Assuming the input dict contains 'obs' key with the observation
        x = input_dict["obs"]
        encoding = self.g(x)
        self.value = self.critic_head(encoding)
        yHat = self.K(encoding)

        print(yHat.shape)
        print(self.value.shape)
        for _ in range(10):
            print("")
        return yHat, state

    def g(self, x):
        # save the original norm
        normX = copy.copy(x)
        normX = normX.reshape(-1, x.shape[-1] * x.shape[-2])
        xNorm = torch.linalg.norm(normX, axis=1)


        # expand the number of channels
        #res = self.inToRes(x)
        res = x #TODO - hack solution, prevents more than 1 res channel from being used. 

        # normal forward pass through res layers
        for layer in self.resLayersG:
            res = layer(res)

        # flatten to vector
        out = res.view(-1, self.inputSize * self.resChannels)

        # expand to encoding dimension
        encodings = self.outToEncoding(out)

        # normalize to input norm
        encodings = encodings.T / torch.linalg.norm(encodings, axis=1)
        encodings = (encodings * xNorm).T
        return encodings

    def gInv(self, encoding):
        # save the original norm
        #flattened = encoding.view(-1,encoding.shape[-2] * encoding.shape[-1]) # flatten this to get the norms
        encodingNorm = torch.linalg.norm(encoding, axis=1)

        # compress from encoding dimension to image size
        out = self.encodingToOut(encoding)
        res = out.view(-1, self.resChannels, self.inputChannels1D + self.inputChannels2D)

        for layer in self.resLayersGInv:
            res = layer(res)

        # reduce the number of channels
        x = self.resToIn(res)
        # ensure that g(x) is a rotation-only operator
        # by restoring the output to its original norm
        x = x.reshape(-1, x.shape[-1] * x.shape[-2] * x.shape[-3])
        xNorm = torch.linalg.norm(x, axis=1)
        x = x.T / xNorm
        x = (x * encodingNorm).T

        x = x.reshape(-1, 1, self.inputChannels1D + self.inputChannels2D)

        return x
    
    @override(TorchModelV2)
    def value_function(self):
        assert self.value is not None, "must call forward first!"
        # print(torch.reshape(self.value, [-1]))
        # for i in range(5):
        #     print("")
        return torch.reshape(self.value, [-1]) #reshape into a single dimension

