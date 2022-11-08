import math
import numpy as np
import torch
import torch.nn as nn

class Embeddings(nn.Module):
  '''
    Args: Size of the vocabulary and dim of the embedding vector
    Return: Word Embeddings
  '''
  def __init__(self,vocab_size, d_model, padding_idx):
    super().__init__()
    self.d_model = d_model
    self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

  def forward(self,x):
    # shape of x : [B x seq_len]
    embeds = self.embedding(x) # shape: [B x seq_len x d_model]
    return embeds * math.sqrt(self.d_model)