import math
import numpy as np
import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from residual_layernorm import ResidualLayerNorm
from positionwise_feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
  '''
    > multi-head attention
    > layer normalization
    > position wise feedforward layer
    > layer normalization
  '''

  def __init__(self, d_model, num_heads, ff_d, dropout=0.3):
    super().__init__()

    self.mha = MultiHeadAttention(d_model, num_heads, dropout)
    self.layer_norm_1 = ResidualLayerNorm(d_model, dropout)
    self.layer_norm_2 = ResidualLayerNorm(d_model, dropout)
    self.ff = PositionwiseFeedForward(d_model, ff_d, dropout)

  def forward(self, x, mask):
    # x: [B x seq x D]
    output, attention_weights = self.mha(x, x, x, mask=mask)
    norm_1 = self.layer_norm_1(output, x)

    ff = self.ff(norm_1)
    norm_2 = self.layer_norm_2(ff, norm_1)

    return norm_2, attention_weights
