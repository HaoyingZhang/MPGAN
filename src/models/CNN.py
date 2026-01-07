import torch
import torch.nn as nn
import torch.nn.functional as F

### --- GENERATOR --- ###
class Generator(nn.Module):
    def __init__(self, n, m, hidden=64):
        super(Generator, self).__init__()
        L = n - m + 1

        input_size = (n - m + 1) * (n - m + 1)
        output_size = n

        self.net = nn.Sequential(
            nn.Conv1d(L, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(n),   # force output length = n
            nn.Conv1d(hidden, 1, kernel_size=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.net(x)
        # y = torch.sigmoid(y)
        
        return y.squeeze(1)



### --- DISCRIMINATOR --- ###
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))  # use last timestep