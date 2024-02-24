from flair.data import Corpus
from flair.datasets import SentenceDataset
from flair.trainers import ModelTrainer
from flair.models import TARSClassifier
from flair.data import Sentence, Label

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re
import copy
from tqdm import tqdm
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

final_texts = []
final_labels = []

def clean_abstract(text):
    # Replace versions with the word "VERSION"
    version_pattern = r"\d+(\.\d+)+"
    updated_text = re.sub(version_pattern, "version", text)

    # Replace other CVE references
    version_pattern = r'CVE-\d{1,5}-\d{1,5}'
    updated_text = re.sub(version_pattern, "CVE", updated_text)

    return updated_text

def get_texts(df):
    texts = df['Text'].apply(clean_abstract)
    texts = texts.values.tolist()
    return texts

def get_labels(df):
    labels_li = [' '.join(x.lower().split()) for x in df.columns.to_list()[2:]]
    labels_matrix = np.array([labels_li] * len(df))

    mask = df.iloc[:, 2:].values.astype(bool)
    labels = []
    for l, m in zip(labels_matrix, mask):
        x = l[m]
        if len(x) > 0:
            labels.append(x.tolist())
        else:
            labels.append([])
    return labels

def get_ids(df):
    ids = df['ID'].values.tolist()
    return ids

def main():
    project_dir = './'
    
    gc.collect()
    torch.cuda.empty_cache()

    df_train = pd.read_csv(project_dir + 'train_1.csv')
    print(df_train.shape)

    df_val = pd.read_csv(project_dir + 'val_1.csv')
    print(df_val.shape)

    df_test = pd.read_csv(project_dir + 'test_1.csv')
    print(df_test.shape)

    train_texts = get_texts(df_train)
    train_labels = get_labels(df_train)
    train_ids = get_ids(df_train)

    val_texts = get_texts(df_val)
    val_labels = get_labels(df_val)
    val_ids = get_ids(df_val)

    test_texts = get_texts(df_test)
    test_labels = get_labels(df_test)
    test_ids = get_ids(df_test)
    #######################################

    tars_train_dataset = []
    train_size = len(train_texts)

    for i in range(train_size):
      sent = Sentence(train_texts[i])

      for label in train_labels[i]:
        sent.add_label(value=label, typename='cybersecurity-attack-tactic')

      tars_train_dataset.append(sent)

    #######################################

    tars_test_dataset = []
    test_size = len(test_texts)

    for i in range(test_size):
      sent = Sentence(test_texts[i])

      for label in test_labels[i]:
        sent.add_label(value=label, typename='cybersecurity-attack-tactic')

      tars_test_dataset.append(sent)

    #######################################

    tars_val_dataset = []
    val_size = len(val_texts)

    for i in range(val_size):
      sent = Sentence(val_texts[i])

      for label in val_labels[i]:
        sent.add_label(value=label, typename='cybersecurity-attack-tactic')

      tars_val_dataset.append(sent)

    #######################################
    
    corpus = Corpus(train=tars_train_dataset, test=tars_val_dataset)

    torch.cuda.empty_cache()

    tars = TARSClassifier.load('tars-base')

    tars.add_and_switch_to_new_task("vulnerability-tactic-prediction", label_type='cybersecurity-attack-tactic', label_dictionary=corpus.make_label_dictionary(label_type='cybersecurity-attack-tactic'))

    trainer = ModelTrainer(tars, corpus)

    trainer.train(base_path='resources/taggers/tactic',
                  learning_rate=0.02,
                  mini_batch_size=1,
                  max_epochs=10,
                  train_with_dev=True,
                  main_evaluation_metric=("weighted avg", "f1-score")
                  )
    
    for i in range(0, val_size):
        sentence = Sentence()

if __name__ == "__main__":
    main()
