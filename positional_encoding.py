import math
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
  '''
    Args: Dim of the embedding vector and Maximum allowed sequence length 
    Formula used in paper: 
        PE(even indexes(2i)) = sin(pos/10000^(2i/d_model))
        PE(odd indexes(2i+1)) = cos(pos/10000^((2i+1)/d_model))
    Return: Positional Encoding
  '''

  def __init__(self, d_model, max_seq_len=512, dropout=0.3, device="cpu"):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    pe = torch.zeros(max_seq_len, d_model).to(device)
    pos = torch.arange(0, max_seq_len).unsqueeze(
        1).float()  # shape: [max_seq_len,1]

    two_i = torch.arange(0, d_model, step=2).float()
    # denominator in the formula
    deno_ = torch.pow(10000, (two_i/torch.tensor([d_model]))).float()

    pe[:, ::2] = torch.sin(pos/deno_)
    pe[:, 1::2] = torch.cos(pos/deno_)

    pe = pe.unsqueeze(0)  # to match batch size

    self.register_buffer("pe", pe)  # saved as non learnable parameters

  def forward(self, x):
    # shape: [B x seq_len x D]
    pe = self.pe[:, :x.shape[1], :].detach()
    pe = pe.repeat(x.shape[0], 1, 1)  # repeated for every batch sample
    x = x.add(pe)
    return self.dropout(x)
