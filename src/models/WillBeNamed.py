import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    """Feature-wise Linear Modulation from a conditioning vector c -> (gamma, beta)."""
    def __init__(self, c_dim, hidden_dim, n_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * n_channels)  # -> gamma, beta
        )
        # init last layer small to start near identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, c):
        gb = self.net(c)              # [B, 2*C]
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma.unsqueeze(-1), beta.unsqueeze(-1)  # [B,C,1] each


class ResDilatedBlock(nn.Module):
    """Residual 1D conv block with dilation + GroupNorm + GELU."""
    def __init__(self, channels, dilation, film: FiLM | None = None, p_drop: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.film = film
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x, gamma=None, beta=None):
        # x: [B,C,L]
        h = self.norm1(x)
        if self.film is not None and gamma is not None and beta is not None:
            h = h * (1 + gamma) + beta
        h = self.act(h)
        h = self.conv1(h)

        h2 = self.norm2(h)
        if self.film is not None and gamma is not None and beta is not None:
            h2 = h2 * (1 + gamma) + beta
        h2 = self.act(h2)
        # h2 = self.drop(h2)
        h2 = self.conv2(h2)
        return x + h2  # residual


class SelfAttention1D(nn.Module):
    """Multi-head self-attention over sequence length (channels as features)."""
    def __init__(self, channels, num_heads=4, proj_drop=0.1):
        super().__init__()
        self.proj_in  = nn.Conv1d(channels, channels, kernel_size=1)
        self.attn     = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)
        self.ln = nn.LayerNorm(channels)
        self.drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: [B,C,L] -> [B,L,C] for attention
        h = self.proj_in(x)
        h = h.transpose(1, 2)          # [B,L,C]
        h = self.ln(h)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = h + attn_out               # residual
        h = h.transpose(1, 2)          # [B,C,L]
        h = self.proj_out(h)
        # h = self.drop(h)
        return x + h                   # residual


class Generator(nn.Module):
    """
    Conditional time-series generator:
      - Input: MP pair(s). Preferred shape [B, L, 2] (dist, index). L = n - m + 1
      - Optional latent z and/or class condition y
      - Output: [B, n] in [0,1]
    """
    def __init__(
        self,
        n: int,
        m: int,
        mp_channels: int = 2,          # MP distance + index
        base_channels: int = 64,
        num_blocks: int = 6,
        dilations: tuple = (1, 2, 4, 8, 16, 32),
        use_attention: bool = True,
        z_dim: int | None = 64,        # set None to disable latent conditioning
        y_dim: int | None = None,      # set to num_classes for label conditioning; None to disable
        film_hidden: int = 128,
        p_drop=0.1,
        attn_drop=0.1,
        proj_drop=0.1
    ):
        super().__init__()
        assert num_blocks == len(dilations), "num_blocks must match length of dilations"
        self.n = n
        self.m = m
        self.L = n - m + 1
        self.z_dim = z_dim
        self.y_dim = y_dim
        cond_dim = (z_dim or 0) + (y_dim or 0)

        # Input adapter: project MP features to channel space
        self.in_proj = nn.Conv1d(mp_channels, base_channels, kernel_size=3, padding=1)

        # FiLM conditioner (optional)
        self.film = FiLM(cond_dim, film_hidden, base_channels) if cond_dim > 0 else None

        # Stack of dilated residual blocks (TCN-style)
        self.blocks = nn.ModuleList([
            ResDilatedBlock(base_channels, d, film=(self.film is not None), p_drop=p_drop)
            for d in dilations
        ])

        # Optional attention
        self.attn = SelfAttention1D(base_channels, num_heads=4, proj_drop=proj_drop) if use_attention else None

        # Head: keep length L, then interpolate to n, then final conv -> 1 channel, sigmoid
        self.mid_norm = nn.GroupNorm(8, base_channels)
        self.mid_act  = nn.GELU()
        self.head     = nn.Conv1d(base_channels, 1, kernel_size=3, padding=1)
        self.drop = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # He/Kaiming init for ReLU-family (fine for GELU)
                nn.init.kaiming_normal_(m.weight, a=0.0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Xavier for fully-connected layers is a solid default
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _prep_input(self, x):
        """
        Accepts:
          - [B, L, 2] (preferred) -> permute to [B, 2, L]
          - OR [B, 2*L] -> reshape to [B, 2, L]
        """
        if x.dim() == 3 and x.size(-1) == 2:         # [B,L,2]
            x = x.permute(0, 2, 1)                   # -> [B,2,L]
        elif x.dim() == 2 and x.size(1) == 2 * self.L:  # [B, 2*L]
            x = x.view(x.size(0), 2, self.L)         # -> [B,2,L]
        elif x.dim() == 3 and x.size(1) == 2:        # already [B,2,L]
            pass
        else:
            raise ValueError(f"Unexpected MP input shape {tuple(x.shape)}. "
                             f"Expected [B,L,2] or [B, 2*L] or [B,2,L] with L={self.L}.")
        return x

    def _build_condition(self, z=None, y=None):
        conds = []
        if self.z_dim is not None:
            if z is None:
                raise ValueError("z_dim set but no z provided")
            conds.append(z)
        if self.y_dim is not None:
            if y is None:
                raise ValueError("y_dim set but no y provided")
            # If y is class indices -> one-hot
            if y.dim() == 1:  # [B]
                y = F.one_hot(y, num_classes=self.y_dim).float()
            conds.append(y)
        if not conds:
            return None
        return torch.cat(conds, dim=-1)  # [B, cond_dim]

    def forward(self, mp_input, z=None, y=None):
        """
        mp_input: [B, L, 2] or [B, 2*L] or [B, 2, L]
        z:       [B, z_dim] (optional)
        y:       [B] (class ids) or [B, y_dim] (one-hot/embedding)
        returns: [B, n] in [0,1]
        """
        B = mp_input.size(0)
        x = self._prep_input(mp_input)            # [B,2,L]
        h = self.in_proj(x)                       # [B,C,L]

        gamma = beta = None
        if self.film is not None:
            c = self._build_condition(z=z, y=y)   # [B, cond_dim]
            gamma, beta = self.film(c)            # [B,C,1] each

        # Dilated residual stack
        for blk in self.blocks:
            h = blk(h, gamma, beta)               # [B,C,L]

        # Optional attention over length L
        if self.attn is not None:
            h = self.attn(h)                      # [B,C,L]

        # Mid processing
        h = self.mid_norm(h)
        if gamma is not None:
            h = h * (1 + gamma) + beta
        h = self.mid_act(h)

        # Interpolate length L -> n smoothly
        if h.size(-1) != self.n:
            h = F.interpolate(h, size=self.n, mode='linear', align_corners=False)  # [B,C,n]
        h = self.drop(h)
        # Head to 1 channel and squash to [0,1]
        y_hat = self.head(h).squeeze(1)           # [B,n]
        # y_hat = torch.sigmoid(y_hat)              # scale to [0,1]; swap to tanh if your target is [-1,1]
        return y_hat
