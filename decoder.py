import math
import numpy as np
import torch
import torch.nn as nn
from decoder_layer import DecoderLayer
from positional_encoding import PositionalEncoding

class Decoder(nn.Module):
  """Combining all decoder layers into a decoder"""

  def __init__(self, Embedding, d_model, ff_d, max_seq_len, num_heads, num_layers, dropout=0.3, device="cpu"):
    super().__init__()
    self.embedding = Embedding
    self.PE = PositionalEncoding(
        d_model, max_seq_len=max_seq_len, device=device)

    self.dropout = nn.Dropout(dropout)
    self.decoders = nn.ModuleList([DecoderLayer(d_model, num_heads, ff_d, dropout)
                                  for _ in range(num_layers)])

  def forward(self, x, encoder_outputs, trg_mask, src_mask):
    # [B x seq_len]

    embeddings = self.embedding(x)
    encoding = self.PE(embeddings)

    for decoder in self.decoders:
      encoding, masked_attention_weights, enc_dec_attention_weights = decoder(
          encoding, encoder_outputs, trg_mask, src_mask)

    return encoding, masked_attention_weights, enc_dec_attention_weights
