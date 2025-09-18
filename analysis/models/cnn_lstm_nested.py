# models/cnn_lstm_model.py

import torch
import torch.nn as nn
import torchvision.models as models

class NestedCNNLSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=1, num_classes=2):
        super(NestedCNNLSTM, self).__init__()

        # Use pretrained ResNet18 as CNN encoder
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # remove FC and AvgPool layers

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.feature_dim = resnet.fc.in_features  # 512 for ResNet18

        # LSTM processes sequence of CNN-extracted features
        self.lstm = nn.LSTM(self.feature_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        cnn_feats = []

        for t in range(T):
            xt = self.cnn(x[:, t])          # (B, 512, H’, W’)
            xt = self.avgpool(xt)           # (B, 512, 1, 1)
            xt = self.flatten(xt)           # (B, 512)
            cnn_feats.append(xt)

        cnn_feats = torch.stack(cnn_feats, dim=1)  # (B, T, 512)

        lstm_out, _ = self.lstm(cnn_feats)         # (B, T, hidden)
        final_feat = lstm_out[:, -1, :]            # (B, hidden)

        return self.fc(final_feat)                 # (B, num_classes)
