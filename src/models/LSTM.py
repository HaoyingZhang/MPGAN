import torch
import torch.nn as nn


### --- GENERATOR --- ###
class Generator(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_length=100):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)  # attention weights
        self.fc = nn.Linear(hidden_dim * 2, output_length)

    def forward(self, mp_input):  # shape: (B, T, 2)
        out, _ = self.lstm(mp_input)       # (B, T, H)
        attn_weights = torch.softmax(self.attention(out), dim=1)  # (B, T, 1)
        context = torch.sum(out * attn_weights, dim=1)            # (B, H)
        out = self.fc(context)                                    # (B, output_length)
        out = torch.sigmoid(out)                                  # (B, output_length)
        return out.unsqueeze(-1)                                  # (B, n, 1)

### --- DISCRIMINATOR --- ###
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))  # use last timestep