import math
import numpy as np
import torch
import torch.nn as nn
from transformer import Transformer
from torch.utils.data import Dataset

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class English2Hindi(nn.Module):
  """Using encoder-decoder system of transformer for machine translation purpose"""

  def __init__(self, src_vocab, trg_vocab, d_model=512, ff_d=2048, max_seq_len=512, num_layers=6, num_heads=8, dropout=0.3, device="cpu"):
    super().__init__()

    self.model = Transformer(len(src_vocab),
                             len(trg_vocab),
                             d_model,
                             ff_d,
                             max_seq_len,
                             num_heads,
                             num_layers,
                             src_pad_idx=src_vocab.__getitem__("<pad>"),
                             trg_pad_idx=trg_vocab.__getitem__("<pad>"),
                             dropout=dropout,
                             device=device
                             )

    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.d_model = d_model

    self.train_losses = AverageMeter()
    self.valid_losses = AverageMeter()

    self.device = device

  def loss(self, outputs, targets):
    loss_fn = nn.CrossEntropyLoss(ignore_index=self.trg_vocab.__getitem__("<pad>"))
    loss = loss_fn(
        outputs.view(-1, outputs.shape[-1]), targets.contiguous().view(-1))
    return loss

  def optimizer_scheduler(self):
    optimizer_params_ = list(self.model.named_parameters())
    no_decay = ["bias", "layer_norm"]
    optimizer_parameters = [
        {
            "params": [p for n, p in optimizer_params_ if not any(nd in n.split(".")[-1] or nd in n.split(".")[-2] for nd in no_decay)],
            "weight_decay": 0.001,
        },
        {
            "params": [p for n, p in optimizer_params_ if any(nd in n.split(".")[-1] or nd in n.split(".")[-2] for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 1e-4, 3e-3, cycle_momentum=False)
    return optimizer, scheduler

  def forward(self, src, trg):
    return self.model(src, trg)


def train_step(model, train_loader, optimizer, scheduler, batch_size=12, gradient_accumulation_steps=1, max_grad_norm=None, device="cpu"):
  model.train()
  scaler = torch.cuda.amp.GradScaler(enabled=True)
  train_losses = AverageMeter()
  preds = []
  for step, batch in enumerate(train_loader):
    src = batch["src"].to(device)
    trg = batch["trg"].to(device)

    trg = trg[:, :-1]  # predicting the next word, so last word not required
    trg = trg[:, 1:]  # words we need to predict

    with torch.cuda.amp.autocast(enabled=True):
      logits = model(src, trg)

    loss = model.loss(logits, trg)
    if gradient_accumulation_steps > 1:
      loss = loss / gradient_accumulation_steps

    if max_grad_norm != None:
      nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    train_losses.update(loss.item(), batch_size)
    scaler.scale(loss).backward()

    if (step+1) % gradient_accumulation_steps == 0 or (step+1) == len(train_loader):
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad()
      scheduler.step()

    preds.append(logits)

  return train_losses.avg, preds


def valid_step(model, val_loader, device="cpu"):
  model.eval()
  val_losses = AverageMeter()
  preds = []
  for step, batch in enumerate(val_loader):
    src = batch["src"].to(device)
    trg = batch["trg"].to(device)

    trg = trg[:, :-1]
    trg = trg[:, 1:]

    with torch.no_grad():
      logits = model(src, trg)

    loss = model.loss(logits, trg)

    val_losses.update(loss.item(), len(val_loader))

    preds.append(logits)

  return val_losses.avg, preds

class English2HindiDataset(Dataset):

  def __init__(self, df, max_seq_len):
    super().__init__()
    self.df = df
    self.max_seq_len = max_seq_len

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    src = []
    eng = self.df["english_sentence"][idx]
    for word in eng.split():
      src.append(word)

    trg = []
    hindi = self.df["hindi_sentence"][idx]
    for word in hindi.split():
      trg.append(word)

    if len(src) > self.max_seq_len:
      src = src[:self.max_seq_len]

    if len(trg) > self.max_seq_len:
      trg = trg[:self.max_seq_len]

    data = {
        "src": src,
        "trg": trg
    }
    return data

class Collate:
  def __init__(self, max_seq_len, src_vocab, trg_vocab):
    self.max_seq_len = max_seq_len
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab

  def __call__(self, batch):
    output = dict()
    src_batch = [sample["src"] for sample in batch]
    trg_batch = [sample["trg"] for sample in batch]

    batch_max = max([len(sent) for sent in src_batch] + [len(sent)
                    for sent in trg_batch])

    if batch_max > self.max_seq_len:
      batch_max = self.max_seq_len

    src_vocab = self.src_vocab
    trg_vocab = self.trg_vocab
    src_batch = [src_vocab(["<sos>"] + s + ["<pad>"] *
                          (batch_max - len(s)) + ["<eos>"]) for s in src_batch]
    trg_batch = [trg_vocab(["<sos>"] + s + ["<pad>"] *
                          (batch_max - len(s)) + ["<eos>"]) for s in trg_batch]

    output["src"] = torch.tensor(src_batch, dtype=torch.long)
    output["trg"] = torch.tensor(trg_batch, dtype=torch.long)

    return output


def token_ids_to_sentences(ids, vocab):
    tokens = vocab.lookup_tokens(ids.tolist())
    tokens = [token for token in tokens if token not in [
        '<sos>', '<pad>', '<eos>', '<unk>']]
    sentence = " ".join(tokens)
    return sentence
