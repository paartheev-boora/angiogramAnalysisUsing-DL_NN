import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_layers=1, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x_seq):
        _, (h_n, _) = self.lstm(x_seq)
        h_last = h_n[-1]
        out = self.fc(h_last)
        return out.squeeze(1)
