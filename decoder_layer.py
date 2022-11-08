import math
import numpy as np
import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from residual_layernorm import ResidualLayerNorm
from positionwise_feedforward import PositionwiseFeedForward
from residual_layernorm import ResidualLayerNorm


class DecoderLayer(nn.Module):
  '''
  > masked multi-head attention
  > layer normalization
  > encoder-decoder multi-head attention (keys and values from encoder)
  > layer normalization
  > position wise feed forward layer
  > layer normalization
  '''

  def __init__(self, d_model, num_heads, ff_d, dropout):
    super().__init__()
    self.masked_mha = MultiHeadAttention(d_model, num_heads, dropout)
    self.layer_norm_1 = ResidualLayerNorm(d_model, dropout)

    self.enc_dec_mha = MultiHeadAttention(d_model, num_heads, dropout)
    self.layer_norm_2 = ResidualLayerNorm(d_model, dropout)

    self.ff = PositionwiseFeedForward(d_model, ff_d)
    self.layer_norm_3 = ResidualLayerNorm(d_model, dropout)

  def forward(self, x, encoder_outputs, trg_mask, src_mask):
    masked_mha_outputs, masked_attention_weights = self.masked_mha(
        x, x, x, mask=trg_mask)
    norm_1 = self.layer_norm_1(masked_mha_outputs, x)

    enc_dec_outputs, enc_dec_attention_weights = self.enc_dec_mha(
        norm_1, encoder_outputs, encoder_outputs, mask=src_mask)  # keys and values are from encoder layer
    norm_2 = self.layer_norm_2(enc_dec_outputs, norm_1)

    ff = self.ff(norm_2)
    norm_3 = self.layer_norm_3(ff, norm_2)

    return norm_3, masked_attention_weights, enc_dec_attention_weights
