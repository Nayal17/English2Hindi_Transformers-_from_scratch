import re
import string
import numpy as np
import pandas as pd

def preprocess(df):
  # removing null values from source sentence
  df = df[~pd.isnull(df["english_sentence"])]

  # removing duplicates
  df = df.drop_duplicates()

  # lowercase english sentence
  df["english_sentence"] = df["english_sentence"].apply(lambda x: x.lower())

  # removing puctuations
  punctuations = set(string.punctuation)
  punctuations.add('।')
  df["english_sentence"] = df["english_sentence"].apply(
      lambda x: ''.join(ch for ch in str(x) if ch not in punctuations))
  df["hindi_sentence"] = df["hindi_sentence"].apply(
      lambda x: ''.join(ch for ch in str(x) if ch not in punctuations))

  # removing digits
  df["english_sentence"] = df["english_sentence"].apply(
      lambda x: x.translate(str.maketrans('', '', '0123456789')))
  df["hindi_sentence"] = df["hindi_sentence"].apply(
      lambda x: x.translate(str.maketrans('', '', '0123456789')))
  df["hindi_sentence"] = df["hindi_sentence"].apply(
      lambda x: x.translate(str.maketrans('', '', '२३०८१५७९४६')))

  # removing extra spaces
  df['english_sentence'] = df['english_sentence'].apply(lambda x: x.strip())
  df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: x.strip())
  df['english_sentence'] = df['english_sentence'].apply(
      lambda x: re.sub(" +", " ", x))
  df['hindi_sentence'] = df['hindi_sentence'].apply(
      lambda x: re.sub(" +", " ", x))
  df = df.reset_index(drop=True)

  return df


def tokenizer(df, src=False):
  if src:
    corpus = df["english_sentence"]
  else:
    corpus = df["hindi_sentence"]

  tokens = []
  for sent in corpus:
    for word in sent.split():
      tokens.append(word)

    yield tokens
