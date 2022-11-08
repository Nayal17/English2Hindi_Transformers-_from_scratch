import math
import numpy as np
import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
  '''
  Position Wise Feed Forward Network
  FNN(x) = max(0,W1*x)*W2 + b2
  shape(w1): d_model x ff_d(2048 on paper)
  shape(w2): ff_d x d_model
  '''

  def __init__(self, d_model, ff_d, dropout=0.3):
    super().__init__()
    self.ffn = nn.Sequential(
        nn.Linear(d_model, ff_d),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(ff_d, d_model)
    )

  def forward(self, x):
    output = self.ffn(x)
    return output
