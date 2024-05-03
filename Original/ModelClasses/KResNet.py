import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import copy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ResNetLayer(nn.Module):

    def __init__(self, nChannels=1, activation="relu", batchNorm=False, dropoutRate=0):
        super(ResNetLayer, self).__init__()

        activationName = activation
        self.batchnorm = batchNorm
        self.nChannels = nChannels

        self.dropout = nn.Dropout(dropoutRate)

        self.batchnormLayer = nn.BatchNorm2d(self.nChannels)

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
        if self.batchnorm:
            return x + self.batchnormLayer(self.conv1(self.dropout(self.activation(self.conv1(self.dropout(x))))))
        else:
            return x + self.conv1(self.dropout(self.activation(self.conv1(self.dropout(x)))))


class KoopmanResNet(nn.Module):
    def __init__(self, nLayers=20, categories=10, encodingDims=1000, inChannels=1, resChannels=10, dropout=0, batchNorm=False, activation="relu",
                 initialization="xh"):
        super(KoopmanResNet, self).__init__()

        params = {"actor_state_size":3, "env_state_size":0, "output_size":1,
                                   "nonspatial_depth":6, "nonspatial_width":128, "output_type":"distribution"}
        nLayers = params["nonspatial_depth"]
        out_dim = params["output_size"]
        self.resChannels = params[""]
        encodingDims = params["nonspatial_width"]



        self.resChannels = resChannels
        self.inputSize = 28 * 28

       # inChannels to resChannels (and vise-versa)
        self.inToRes = nn.Conv2d(in_channels=inChannels, out_channels=resChannels, kernel_size=3, padding=1)
        self.resToIn = nn.Conv2d(in_channels=resChannels, out_channels=inChannels, kernel_size=3, padding=1)
        # resLayers (and vis-versa)
        self.resLayersG = nn.ModuleList([])
        for layer in range(nLayers):
            self.resLayersG.append(ResNetLayer(nChannels=resChannels, activation=activation, batchNorm=batchNorm,dropoutRate=dropout))

        self.resLayersGInv = nn.ModuleList([])
        for layer in range(nLayers):
            self.resLayersGInv.append(ResNetLayer(nChannels=resChannels, activation=activation, batchNorm=batchNorm,dropoutRate=dropout))

        # outChannels to categories (and vise-versa)
        self.outToEncoding = nn.Linear(in_features=self.inputSize * resChannels, out_features=encodingDims)
        self.encodingToOut = nn.Linear(in_features=encodingDims, out_features=self.inputSize * resChannels)

        self.K = nn.Linear(encodingDims, categories, bias=False)

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

    def forward(self, x):
        yHat = self.K(self.g(x))
        return yHat

    def g(self, x):
        # save the original norm
        normX = copy.copy(x)
        normX = normX.reshape(-1, x.shape[-1] * x.shape[-2] * x.shape[-3])
        xNorm = torch.linalg.norm(normX, axis=1)

        # normal forward pass through res layers

        # expand the number of channels
        res = self.inToRes(x)

        # residual processing
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
        res = out.view(-1, self.resChannels, 28, 28)

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

        x = x.reshape(-1, 1, 28, 28)

        return x

