import math
import numpy as np
import torch
import torch.nn as nn
from embeddings import Embeddings
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
  def __init__(self, src_vocab_len, trg_vocab_len, d_model, ff_d, max_seq_len, num_heads,
               num_layers, src_pad_idx, trg_pad_idx, dropout=0.3, device="cpu"):
    super().__init__()

    self.src_pad_idx = src_pad_idx
    self.trg_pad_idx = trg_pad_idx

    encoder_embedding = Embeddings(
        src_vocab_len, d_model, src_pad_idx)

    decoder_embedding = Embeddings(
        trg_vocab_len, d_model, trg_pad_idx)

    self.encoder = Encoder(encoder_embedding, d_model, ff_d,
                           max_seq_len, num_heads, num_layers, dropout, device=device)
    self.decoder = Decoder(decoder_embedding, d_model, ff_d,
                           max_seq_len, num_heads, num_layers, dropout, device=device)

    self.linear = nn.Linear(d_model, trg_vocab_len)
    self.device = device

    for p in self.parameters():
      if p.dim() > 1:  # fan_in and fan_out (in xavier uniform) cannot be calculated for fewer than 2 dims.
        nn.init.xavier_uniform_(p)

  def create_src_mask(self, src):
    mask = (src != self.src_pad_idx).unsqueeze(1).to(self.device)
    return mask

  def create_trg_mask(self, trg):
    trg_mask = (trg != self.trg_pad_idx).unsqueeze(1)  # [B x 1 x seq_len]
    mask = torch.ones((trg.shape[1], trg.shape[1])).tril(0).to(
        self.device)  # [1 x seq_len x seq_len] , mask will be same for all
    mask = mask != 0  # converting to boolean , True where index value is not 0
    trg_mask = trg_mask & mask  # [B x seq_len x seq_len]
    return trg_mask

  def forward(self, src, trg):
    # [B x seq_len]

    src_mask = self.create_src_mask(src)  # [B x 1 x seq_len]
    trg_mask = self.create_trg_mask(trg)  # [B x seq_len x seq_len]

    encoder_outputs, encoder_attention_weights = self.encoder(
        src, mask=src_mask)
    decoder_outputs, masked_attention_weights, enc_dec_attention_weights = self.decoder(
        trg, encoder_outputs, trg_mask, src_mask)

    logits = self.linear(decoder_outputs)  # [B x trg_seq_len x trg_vocab_size]

    return logits
