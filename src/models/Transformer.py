import math
import torch
import torch.nn as nn


# ---------------------------
# Positional Encoding
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, d_model]
        return x + self.pe[:x.size(1)].unsqueeze(0)


# ---------------------------
# MP Encoder–Decoder
# ---------------------------
class MPEncoderDecoder(nn.Module):
    """
    Encoder–Decoder Transformer for Matrix Profile inversion

    Input:
        MP embedding: [B, L, L]
    Output:
        Time series:  [B, n, 1]
    """

    def __init__(
        self,
        n: int,
        m: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n = n
        self.L = n - m + 1

        # -------- Encoder --------
        # Project MP rows (L features) → d_model
        self.enc_input_proj = nn.Linear(self.L, d_model)
        self.enc_pos = PositionalEncoding(d_model, max_len=self.L)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        # -------- Decoder --------
        # Learned queries (one per output time step)
        self.query_embed = nn.Embedding(n, d_model)
        self.dec_pos = PositionalEncoding(d_model, max_len=self.n)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        # Output projection
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, mp_embedding):
        """
        mp_embedding: [B, L, L]
        """
        B = mp_embedding.size(0)

        # ---- Encode Matrix Profile ----
        x = self.enc_input_proj(mp_embedding)   # [B, L, d_model]
        x = self.enc_pos(x)
        memory = self.encoder(x)                # [B, L, d_model]

        # ---- Decode time series ----
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        queries = self.dec_pos(queries)         # [B, n, d_model]

        dec_out = self.decoder(
            tgt=queries,
            memory=memory
        )                                        # [B, n, d_model]

        y = self.out_proj(dec_out)              # [B, n, 1]

        return y.squeeze(-1) # [B, n]
