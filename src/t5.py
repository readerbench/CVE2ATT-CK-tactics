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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from transformers import (
    T5Tokenizer,
    T5Model,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import multilabel_confusion_matrix

##############################################################

BEST_F1 = 0
BEST_TRUE = []
BEST_PREDICTED = []

device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
####i##########################################################

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
            labels.append((x.tolist()))

    return labels

"""# Config"""

class Config:
    def __init__(self):
        super(Config, self).__init__()

        self.SEED = 42
        self.MODEL_PATH = 't5-base'

        self.TOKENIZER = T5Tokenizer.from_pretrained(self.MODEL_PATH)
        self.SRC_MAX_LENGTH = 320
        self.TGT_MAX_LENGTH = 20
        self.BATCH_SIZE = 16
        self.VALIDATION_SPLIT = 0.20

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FULL_FINETUNING = True
        self.LR = 2e-4

        self.OPTIMIZER = 'AdamW'
        self.CRITERION = 'BCEWithLogitsLoss'
        self.SAVE_BEST_ONLY = True
        self.N_VALIDATE_DUR_TRAIN = 3
        self.EPOCHS = 5

"""# Dataset & Dataloader"""

class T5Dataset(Dataset):
    def __init__(self, df, set_type=None, custom_type=None):
        super(T5Dataset, self).__init__()

        self.texts = get_texts(df)
        self.set_type = set_type

        labels_df = df['label'].tolist()
        self.labels = [x + ' </s>' for x in labels_df]

        self.tokenizer = Config().TOKENIZER
        self.src_max_length = Config().SRC_MAX_LENGTH
        self.tgt_max_length = Config().TGT_MAX_LENGTH

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        src_tokenized = self.tokenizer.encode_plus(
            self.texts[index],
            max_length=self.src_max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        src_input_ids = src_tokenized['input_ids'].squeeze()
        src_attention_mask = src_tokenized['attention_mask'].squeeze()

        if self.set_type != 'test':
            tgt_tokenized = self.tokenizer.encode_plus(
                self.labels[index],
                max_length=self.tgt_max_length,
                pad_to_max_length=True,
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt'
            )
            tgt_input_ids = tgt_tokenized['input_ids'].squeeze()
            tgt_attention_mask = tgt_tokenized['attention_mask'].squeeze()

            return {
                'src_input_ids': src_input_ids.long(),
                'src_attention_mask': src_attention_mask.long(),
                'tgt_input_ids': tgt_input_ids.long(),
                'tgt_attention_mask': tgt_attention_mask.long()
            }

        return {
            'src_input_ids': src_input_ids.long(),
            'src_attention_mask': src_attention_mask.long()
        }

"""# Model"""

class T5Model(nn.Module):
    def __init__(self):
        super(T5Model, self).__init__()

        self.t5_model = T5ForConditionalGeneration.from_pretrained(Config().MODEL_PATH)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None
        ):

        return self.t5_model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

"""# Engine"""

def compute_metrics(predicted_y, true_y, metric_function, columns, limit):
  results = pd.DataFrame(columns = columns)
  if (metric_function == accuracy_score):
    results.loc[len(results)] = metric_function(true_y, predicted_y)
  else:
    results.loc[len(results)] = metric_function(true_y, predicted_y, average=None)
    
  sorted_results = results.sort_values(by=0, axis=1, ascending=False)

  for col in sorted_results.columns[:limit]:
        print(f"{col}: {sorted_results[col].values[0]}")

  return sorted_results.iloc[:, :limit]

def print_F1_based_on_distribution(y_true, y_pred, Y, columns):
  fig,ax = plt.subplots()

  results = pd.DataFrame(columns = columns)
  results.loc[len(results)] = f1_score(y_true, y_pred, average=None)

  Y_count = Y.apply(np.sum, axis=0)
  Y_count_sorted = Y_count.sort_values(ascending=False)

  ax.bar(Y_count_sorted.index, Y_count_sorted.values)
  ax.set_xlabel("Tactics")
  ax.set_ylabel("Number of CVEs")
  plt.xticks(rotation=90)

  ax2=ax.twinx()
  ax2.plot(Y_count_sorted.index, results[Y_count_sorted.index].iloc[0], color='red')
  ax2.set_ylabel("F1 Score")

  ax = plt.gca()
  plt.show()

def get_ohe(sequences):
    global current_labels

    labels_to_idx = dict()
    label_names = current_labels
    for idx, label in enumerate(label_names):
        labels_to_idx[label] = idx

    ohe = []
    for seq in sequences:
        current_ohe = [0] * 14

        processed_seq = seq.replace("Predicted MITRE tactics:", "")
        processed_seq = processed_seq.replace("</s>", "")
        processed_seq = processed_seq.replace("<pad>", "")
        processed_seq = processed_seq.replace(" ", "")
        labels = processed_seq.split(',')

        for label in labels:
            idx = labels_to_idx.get(label, -1)
            if idx != -1:
                current_ohe[idx] = 1

        ohe.append(current_ohe)

    ohe = np.array(ohe)
    return ohe

def val(model, val_dataloader, criterion, is_final_test=False):
    global BEST_F1, BEST_TRUE, BEST_PREDICTED
    
    print("Val Device:", device)
    val_loss = 0
    true, pred = [], []

    model.eval()

    for step, batch in enumerate(val_dataloader):
        b_src_input_ids = batch['src_input_ids'].to(device)
        b_src_attention_mask = batch['src_attention_mask'].to(device)

        b_tgt_input_ids = batch['tgt_input_ids']
        labels = b_tgt_input_ids.to(device)
        labels[labels[:, :] == Config().TOKENIZER.pad_token_id] = -100

        b_tgt_attention_mask = batch['tgt_attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=b_src_input_ids,
                attention_mask=b_src_attention_mask,
                labels=labels,
                decoder_attention_mask=b_tgt_attention_mask)
            loss = outputs[0]

            val_loss += loss.item()

            for true_id in b_tgt_input_ids:
                true_decoded = Config().TOKENIZER.decode(true_id)
                true.append(true_decoded)

            pred_ids = model.t5_model.generate(
                input_ids=b_src_input_ids,
                attention_mask=b_src_attention_mask,
            )

            pred_ids = pred_ids.cpu().numpy()
            for pred_id in pred_ids:
                pred_decoded = Config().TOKENIZER.decode(pred_id)
                pred.append(pred_decoded)

    true_ohe = get_ohe(true)
    pred_ohe = get_ohe(pred)

    avg_val_loss = val_loss / len(val_dataloader)
    print('Val loss:', avg_val_loss)
    print('Val accuracy:', accuracy_score(true_ohe, pred_ohe))

    print('Val precision:', precision_score(true_ohe, pred_ohe, average='weighted'))
    print('Val recall:', recall_score(true_ohe, pred_ohe, average='weighted'))

    val_micro_f1_score = f1_score(true_ohe, pred_ohe, average='micro')
    print('Val micro f1 score:', val_micro_f1_score)

    val_macro_f1_score = f1_score(true_ohe, pred_ohe, average='macro')
    print('Val macro f1 score:', val_macro_f1_score)

    val_weighted_f1_score = f1_score(true_ohe, pred_ohe, average='weighted')
    print('Val weighted f1 score:', val_weighted_f1_score)

    if (is_final_test is True):
      BEST_F1 = val_weighted_f1_score
      BEST_TRUE = true_ohe
      BEST_PREDICTED = pred_ohe
    elif (val_weighted_f1_score > BEST_F1):
      BEST_F1 = val_weighted_f1_score
      BEST_TRUE = true_ohe
      BEST_PREDICTED = pred_ohe

    return val_weighted_f1_score

def train(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    scheduler,
    epoch
    ):
    nv = Config().N_VALIDATE_DUR_TRAIN
    temp = len(train_dataloader) // nv
    temp = temp - (temp % 100)
    validate_at_steps = [temp * x for x in range(1, nv + 1)]

    train_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader,
                                      desc='Epoch ' + str(epoch))):
        model.train()

        b_src_input_ids = batch['src_input_ids'].to(device)
        b_src_attention_mask = batch['src_attention_mask'].to(device)

        labels = batch['tgt_input_ids'].to(device)
        labels[labels[:, :] == Config().TOKENIZER.pad_token_id] = -100

        b_tgt_attention_mask = batch['tgt_attention_mask'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=b_src_input_ids,
                        attention_mask=b_src_attention_mask,
                        labels=labels,
                        decoder_attention_mask=b_tgt_attention_mask)
        loss = outputs[0]
        train_loss += loss.item()

        loss.backward()

        optimizer.step()

        scheduler.step()

        if step in validate_at_steps:
            print(f'-- Step: {step}')
            _ = val(model, val_dataloader, criterion)

    avg_train_loss = train_loss / len(train_dataloader)
    print('Training loss:', avg_train_loss)

"""# Run"""

def run():
    global train_data, val_data, train_dataloader, val_dataloader, model
    
    torch.manual_seed(Config().SEED)

    criterion = nn.BCEWithLogitsLoss()

    if Config().FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(optimizer_parameters, lr=Config().LR)

    num_training_steps = len(train_dataloader) * Config().EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    max_val_weighted_f1_score = float('-inf')
    for epoch in range(Config().EPOCHS):
        train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epoch)
        val_weighted_f1_score = val(model, val_dataloader, criterion)

        if Config().SAVE_BEST_ONLY:
            if val_weighted_f1_score > max_val_weighted_f1_score:
                best_model = copy.deepcopy(model)

                model_name = 't5_best_model'
                torch.save(best_model.state_dict(), model_name + '.pt')

                print(f'--- Best Model. F1: {max_val_weighted_f1_score} -> {val_weighted_f1_score}')
                max_val_weighted_f1_score = val_weighted_f1_score

    return best_model, max_val_weighted_f1_score

def transform_df_basic(df):
    df = df.rename(columns={'Resource Development': 'Resources'})
    df = df.rename(columns={'Initial Access': 'Access'})
    df = df.rename(columns={'Privilege Escalation': 'Escalation'})
    df = df.rename(columns={'Defense Evasion': 'Evasion'})
    df = df.rename(columns={'Credential Access': 'Credentials'})
    df = df.rename(columns={'Lateral Movement': 'Movement'})
    df = df.rename(columns={'Command and Control': 'Control'})

    return df

def transform_df(df):
    df = transform_df_basic(df)

    new_columns = df.columns[2:]
    new_label = df[new_columns].apply(lambda x: ', '.join(new_columns[x == 1]), axis=1)
    df_extended = pd.DataFrame({'ID': df['ID'], 'Text': "CVE Text: " + df['Text'], 'label': "Predicted MITRE tactics: " + new_label})

    return df_extended

from transformers import logging 
def main():
    global train_data, val_data, train_dataloader, val_dataloader, model, current_labels

    project_dir = './'

    df_train = pd.read_csv(project_dir + 'train_val_data.csv')
    df_basic = transform_df_basic(df_train)
    current_labels = df_basic.columns[2:]
    print("Current labels")
    print(current_labels)

    df_train = transform_df(df_train)

    df_val = pd.read_csv(project_dir + 'train_val_data.csv')
    df_val = transform_df(df_val)

    df_test = pd.read_csv(project_dir + 'test_data.csv')
    df_test = transform_df(df_test)

    logging.set_verbosity_error()

    train_data = T5Dataset(df_train, custom_type="train")
    val_data = T5Dataset(df_val, custom_type="validation")
    test_data = T5Dataset(df_test, custom_type="test")

    train_dataloader = DataLoader(train_data, batch_size=Config().BATCH_SIZE)
    val_dataloader = DataLoader(val_data, batch_size=Config().BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=Config().BATCH_SIZE)

    device = Config().DEVICE
    print("Device:", Config().DEVICE)

    model = T5Model()
    model.to(Config().DEVICE);

    best_model, best_val_weighted_f1_score = run()

    print("------Validation results --------")
    print(BEST_F1)

    print("F1 scores per class")
    f1_best_metrics = compute_metrics(BEST_PREDICTED, BEST_TRUE, f1_score, current_labels, 14)

    print("Recall scores per class")
    recall_best_metrics = compute_metrics(BEST_PREDICTED, BEST_TRUE, recall_score, current_labels, 14)

    print("Precision scores per class")
    precision_best_metrics = compute_metrics(BEST_PREDICTED, BEST_TRUE, precision_score, current_labels, 14)

    print("Accuracy scores per class")
    acc_best_metrics = compute_metrics(BEST_PREDICTED, BEST_TRUE, accuracy_score, current_labels, 14)

    y_train_df = df_basic.drop(df_basic.columns[0:2], axis=1)

    y_train_df.head()

    print_F1_based_on_distribution(BEST_PREDICTED, BEST_TRUE, y_train_df,  current_labels)

    print("------Testing results --------")
    test_weighted_f1_score = val(model, test_dataloader, nn.BCEWithLogitsLoss(), True)

    print("F1 scores")
    print(BEST_F1)

    f1_metrics = compute_metrics(BEST_PREDICTED, BEST_TRUE, f1_score, current_labels, 14)

    print(f1_metrics)

    print("Recall scores")
    recall_metrics = compute_metrics(BEST_PREDICTED, BEST_TRUE, recall_score, current_labels, 14)

    print("Precision scores")
    precision_best_metrics = compute_metrics(BEST_PREDICTED, BEST_TRUE, precision_score, current_labels, 14)

    print("Accuracy scores")
    acc_metrics = compute_metrics(BEST_PREDICTED, BEST_TRUE, accuracy_score, current_labels, 14)

    print_F1_based_on_distribution(BEST_PREDICTED, BEST_TRUE, y_train_df,  current_labels)

    for col in df_test.columns:
        print(col)

    print("Class counts true")
    class_counts = np.sum(BEST_TRUE, axis=0)
    print(class_counts)

    print("Class counts predicted")
    class_counts = np.sum(BEST_PREDICTED, axis=0)
    print(class_counts)

    conf_matrix = multilabel_confusion_matrix(BEST_TRUE, BEST_PREDICTED)

    for i, matrix in enumerate(conf_matrix):
        print(f"Confusion Matrix for Class {i + 1}:\n{matrix}\n")

if __name__ == "__main__":
    main()
