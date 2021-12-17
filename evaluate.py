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

corpus = read_corpus(train_corpus_path)
dct = build_dict(corpus)

# evaluate
test_corpus = read_corpus(test_corpus_path)
test_ds = NER_DataSet(test_corpus, dct)
test_dl = DataLoader(test_ds, batch_size = BATCH_SIZE)

# build dataloader
np.random.shuffle(corpus)
train_ds = NER_DataSet(corpus[:], dct)

train_dl = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True, drop_last = True, num_workers = 0)

# create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BiLSTM_CRF(Config())
model.load_state_dict(torch.load(path.weights_path + path.saved_model_name, map_location = torch.device(device)))

metric_train = evaluate(model, train_dl, device, toast = "predicting training set", verbose = True)
print("\nTrain set")
print(metric_train.report())

metric_test = evaluate(model, test_dl, device, toast = "predicting test set", verbose = True)
print("\nTest set")
print(metric_test.report())
