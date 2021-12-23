import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim

import path
from config import Config, BATCH_SIZE
from data import read_corpus, build_dict, TAG_MAP, NER_DataSet, condtraints
from model import BiLSTM_CRF
from trainer import train, evaluate

train_corpus_path = path.train_file_path
test_corpus_path = path.test_file_path
val_corpus_path = path.val_file_path
submit_corpus_path = "./datasets/yidu_submit.txt"
saved_model_name = 'yidu_bilstm_crf_ep13.pth'

corpus = read_corpus(train_corpus_path)
dct = build_dict(corpus)

# validate
val_corpus = read_corpus(val_corpus_path)
val_ds = NER_DataSet(val_corpus, dct)
val_dl = DataLoader(val_ds, batch_size = BATCH_SIZE)

# evaluate
test_corpus = read_corpus(test_corpus_path)
test_ds = NER_DataSet(test_corpus, dct)
test_dl = DataLoader(test_ds, batch_size = BATCH_SIZE)

# submit
submit_corpus = read_corpus(submit_corpus_path)
submit_ds = NER_DataSet(submit_corpus, dct)
submit_dl = DataLoader(submit_ds, batch_size = BATCH_SIZE)

# create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BiLSTM_CRF(Config())
model.load_state_dict(torch.load(path.weights_path + saved_model_name, map_location = torch.device(device)))

metric_val = evaluate(model, val_dl, device, toast = "predicting validate set", verbose = True)
print("\nValidate set")
print(metric_val.report())

metric_test = evaluate(model, test_dl, device, toast = "predicting test set", verbose = True)
print("\nTest set")
print(metric_test.report())

metric_submit = evaluate(model, submit_dl, device, toast = "predicting test set", verbose = True)
print("\nSubmit set")
print(metric_submit.report())
