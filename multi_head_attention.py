import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads, dropout=0.3):
    super().__init__()
    self.d_model = d_model
    self.dropout = nn.Dropout(dropout)
    # because final representation comes from concatenation of all heads
    self.head_dim = d_model//num_heads

    self.linear_Qs = nn.ModuleList([nn.Linear(d_model, self.head_dim)
                                    for _ in range(num_heads)])
    self.linear_Ks = nn.ModuleList([nn.Linear(d_model, self.head_dim)
                                    for _ in range(num_heads)])
    self.linear_Vs = nn.ModuleList([nn.Linear(d_model, self.head_dim)
                                    for _ in range(num_heads)])

    self.final_linear = nn.Linear(d_model, d_model)

  def scaled_dot_product_attention(self, Q, K, V, mask=None):
    '''
    Q: queries
    K: Keys
    V: Values
    shape: [B x seq_len x head_dim]

    Attention_weights = softmax(Q.K_trans/sqrt(head_dim)) -- shape: [B x seq_len x seq_len]
    output = attention_weights.V -- shape: [B x seq_len x head_dim]
    '''
    QK = torch.matmul(Q, K.permute(0, 2, 1))
    scaled_QK = QK/math.sqrt(self.head_dim)

    if mask is not None:
      scaled_QK = scaled_QK.masked_fill(mask == 0, -1e+4)

    attention_weights = F.softmax(scaled_QK, dim=-1)
    output = torch.matmul(attention_weights, V)

    return attention_weights, output

  def forward(self, q, k, v, mask=None):
    # q, k, v : [B x seq_len x D]
    Q = [layer(q) for layer in self.linear_Qs]
    K = [layer(k) for layer in self.linear_Ks]
    V = [layer(v) for layer in self.linear_Vs]

    att_weights_per_head = []
    output_per_head = []

    for Q_, K_, V_ in zip(Q, K, V):
      attention_weights, output = self.scaled_dot_product_attention(
          Q_, K_, V_, mask=mask)
      att_weights_per_head.append(attention_weights)
      output_per_head.append(output)

    # shape: [B x seq_len x d_model]
    output = torch.cat(output_per_head, dim=-1)
    attention_weights = torch.stack(att_weights_per_head).permute(
        1, 0, 2, 3)  # shape: [B x num_heads x seq_len x seq_len]

    projection = self.dropout(self.final_linear(output))

    return projection, attention_weights
