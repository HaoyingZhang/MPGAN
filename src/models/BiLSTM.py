import torch
import torch.nn as nn
import torch.nn.functional as F


### --- GENERATOR --- ###
class Generator(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_length=100):  # output_length = n
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_length)

    def forward(self, mp_input):  # mp_input: (B, n-m+1, 2)
        out, _ = self.lstm(mp_input)                 # out: (B, n-m+1, hidden_dim)
        out = self.fc(out[:, -1, :])                 # use last hidden state → (B, output_length)
        out = torch.sigmoid(out)                     # ensure values in [0, 1]
        return out.unsqueeze(-1)                     # shape: (B, n, 1)

### --- DISCRIMINATOR --- ###
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))  # use last timestep

class PhaseShuffle(nn.Module):
    def __init__(self, shift_factor: int):
        """
        Randomly shifts the time axis of each sample by k ∈ [-shift_factor, +shift_factor],
        reflect-padding to keep the same length. Automatically caps k so padding is always legal.
        """
        super().__init__()
        self.shift_factor = shift_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shift_factor == 0:
            return x

        batch_size, channels, seq_len = x.shape
        # cap max shift to at most half the sequence length
        max_shift = min(self.shift_factor, seq_len // 2)
        if max_shift == 0:
            return x

        # sample one shift per example
        k_list = torch.randint(-max_shift, max_shift + 1, (batch_size,), device=x.device)
        x_out = x.clone()

        for k in k_list.unique():
            idxs = (k_list == k).nonzero(as_tuple=False).squeeze(1)
            if k > 0:
                # trim last k, pad k on left
                trimmed = x[idxs, :, :-k]
                x_out[idxs] = F.pad(trimmed, (k, 0), mode='reflect')
            elif k < 0:
                # trim first |k|, pad |k| on right
                trimmed = x[idxs, :, -k:]
                x_out[idxs] = F.pad(trimmed, (0, -k), mode='reflect')
            # k == 0 → leave x_out[idxs] unchanged

        return x_out



class Pulse2pulseDiscriminator(nn.Module):
    def __init__(
        self,
        model_size: int = 64,
        num_channels: int = 1,
        shift_factor: int = 2,
        alpha: float = 0.2,
        verbose: bool = False,
    ):
        super().__init__()
        self.model_size = model_size
        self.num_channels = num_channels
        self.alpha = alpha
        self.verbose = verbose

        # conv layers
        self.conv1 = nn.Conv1d(num_channels, model_size, kernel_size=25, stride=2, padding=11)
        self.conv2 = nn.Conv1d(model_size, 2*model_size, kernel_size=25, stride=2, padding=11)
        self.conv3 = nn.Conv1d(2*model_size, 5*model_size, kernel_size=25, stride=2, padding=11)
        self.conv4 = nn.Conv1d(5*model_size, 10*model_size, kernel_size=25, stride=2, padding=11)
        self.conv5 = nn.Conv1d(10*model_size, 20*model_size, kernel_size=25, stride=4, padding=11)
        self.conv6 = nn.Conv1d(20*model_size, 25*model_size, kernel_size=25, stride=4, padding=11)
        # “same” padding for conv7 so kernel_size ≤ padded_length even when seq_len=1
        self.conv7 = nn.Conv1d(25*model_size, 100*model_size, kernel_size=25, stride=4, padding=12)

        # phase‐shuffle layers
        self.ps1 = PhaseShuffle(shift_factor)
        self.ps2 = PhaseShuffle(shift_factor)
        self.ps3 = PhaseShuffle(shift_factor)
        self.ps4 = PhaseShuffle(shift_factor)
        self.ps5 = PhaseShuffle(shift_factor)
        self.ps6 = PhaseShuffle(shift_factor)

        # final linear (size matches 100*model_size * 1 after last conv)
        self.fc1 = nn.Linear(100*model_size * 1, 1)

        # weight init
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, channels, length)
        """
        # conv+leaky+shuffle ×6
        for conv, ps in zip(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6],
            [self.ps1,    self.ps2,    self.ps3,    self.ps4,    self.ps5,    self.ps6],
        ):
            x = F.leaky_relu(conv(x), negative_slope=self.alpha)
            if self.verbose:
                print(x.shape)
            x = ps(x)

        # conv7 + leaky
        x = F.leaky_relu(self.conv7(x), negative_slope=self.alpha)
        if self.verbose:
            print("after conv7:", x.shape)

        # flatten and linear
        x = x.view(x.size(0), -1)
        if self.verbose:
            print("flatten:", x.shape)

        return torch.sigmoid(self.fc1(x))
