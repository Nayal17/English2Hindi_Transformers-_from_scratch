import gc
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

import torch
from torchtext.vocab import build_vocab_from_iterator, Vocab
from preprocessing import preprocess, tokenizer
from utils import English2HindiDataset, English2Hindi, Collate, train_step, valid_step, token_ids_to_sentences

if __name__ == "__main__":
  # hyper_params
  max_seq_len = 40
  device = "cpu"
  batch_size = 12
  gradient_accumulation_steps = 1
  max_grad_norm = 1000
  epochs = 15
  folds = 4
  seed = 69
  SAVE = True

  df = pd.read_csv(
      "./Dataset/Hindi_English_Truncated_Corpus.csv")
  df = preprocess(df)

  # building vocab
  df = df[:10]
  src_vocab = build_vocab_from_iterator(tokenizer(df, src=True), specials=[
                                        "<unk>", "<pad>", "<sos>", "<eos>"])
  src_vocab.set_default_index(src_vocab["<unk>"])
  trg_vocab = build_vocab_from_iterator(tokenizer(df, src=False), specials=[
                                        "<unk>", "<pad>", "<sos>", "<eos>"])
  trg_vocab.set_default_index(trg_vocab["<unk>"])

  kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
  for fold, (train_idx, val_idx) in enumerate(kf.split(df.english_sentence, df.hindi_sentence)):
    train_df = df.loc[train_idx].reset_index(drop=True)
    valid_df = df.loc[val_idx].reset_index(drop=True)

    collate_fn = Collate(max_seq_len, src_vocab, trg_vocab)
    # train data
    train_dataset = English2HindiDataset(train_df, max_seq_len)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # valid data
    val_dataset = English2HindiDataset(valid_df, max_seq_len)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # model
    model = English2Hindi(src_vocab, trg_vocab, device=device)

    # optimizer and scheduler
    optimizer, scheduler = model.optimizer_scheduler()

    best_loss = np.inf
    for epoch in range(epochs):
      print("#"*20 + f" Fold_{fold} |" + f" Epoch_{epoch} " + "#"*20)
      train_loss, train_preds = train_step(
          model, train_loader, optimizer, scheduler, max_grad_norm=max_grad_norm, device=device)
      valid_loss, valid_preds = valid_step(model, val_loader, device=device)

      if valid_loss < best_loss:
        best_loss = valid_loss
        if SAVE:
          torch.save(model.state_dict(), f"./E2H_fold{fold}_best.pth")

      torch.cuda.empty_cache()
      gc.collect()

      print(f"train_loss: {train_loss}, valid_loss: {valid_loss}\n")
