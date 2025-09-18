import torch
import torch.nn as nn
from torchvision import models

class CNNEncoder(nn.Module):
    def __init__(self, output_dim=512, freeze=True):
        super(CNNEncoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # remove FC

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.features(x)  # [B, 512, 1, 1]
        return features.view(features.size(0), -1)  # [B, 512]
