import math
import numpy as np
import torch
import torch.nn as nn
from encoder_layer import EncoderLayer
from positional_encoding import PositionalEncoding

class Encoder(nn.Module):
  """Combining all encoder layers into a encoder"""

  def __init__(self, Embedding, d_model, ff_d, max_seq_len, num_heads, num_layers, dropout=0.3, device="cpu"):
    super().__init__()
    self.embedding = Embedding
    self.PE = PositionalEncoding(d_model, max_seq_len, dropout, device=device)

    self.encoders = nn.ModuleList([EncoderLayer(d_model, num_heads, ff_d, dropout)
                                  for _ in range(num_layers)])
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask=None):
    # [B x seq_len]

    embeddings = self.embedding(x)
    encoding = self.PE(embeddings)

    for encoder in self.encoders:
      encoding, encoder_attention_weights = encoder(encoding, mask)

    return encoding, encoder_attention_weights
