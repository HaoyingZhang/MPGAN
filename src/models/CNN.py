import torch
import torch.nn as nn
import torch.nn.functional as F

### --- GENERATOR --- ###
class Generator(nn.Module):
    def __init__(self, n, m):
        super(Generator, self).__init__()
        self.n = n
        self.m = m

        input_size = 2 * (n - m + 1)
        output_size = n

        # Convolution: input_size -> output_size
        # Kernel size m ensures (n - m + 1) receptive field
        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=output_size, 
            kernel_size=m
        )

        # Fully connected to adjust dimensions if necessary
        self.fc = nn.Linear(n * (input_size - m + 1), output_size)

    def forward(self, x):
        # Expect input shape: (batch_size, input_size)
        x = x.unsqueeze(1)  # add channel dimension: (batch, 1, input_size)

        # Conv1d: (batch, out_channels=n, L_out)
        x = self.conv(x)

        # Non-linearity
        x = F.relu(x)

        # Flatten: (batch, n, L_out) -> (batch, n*L_out)
        x = x.view(x.size(0), -1)

        # Project to output size n
        x = self.fc(x)

        x = torch.sigmoid(x)
        
        return x



### --- DISCRIMINATOR --- ###
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))  # use last timestep