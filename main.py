##
import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import pytorch_model_summary as pms

import path
from config import Config, BATCH_SIZE
from data import read_corpus, build_dict, NER_DataSet
from model import BiLSTM_CRF
from trainer import train, evaluate

train_corpus_path = path.train_file_path
val_corpus_path = path.val_file_path
test_corpus_path = path.test_file_path


def plot(history):
    plt.rcParams['font.sans-serif'] = ['SF Mono']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 360  # 图片像素
    plt.rcParams['figure.dpi'] = 360  # 分辨率
    train_process = pd.DataFrame(history)
    
    # 设置value的显示长度为200，默认为50
    pd.set_option('max_colwidth', 200)
    # 显示所有列，把行显示设置成最大
    pd.set_option('display.max_columns', None)
    # 显示所有行，把列显示设置成最大
    pd.set_option('display.max_rows', None)
    
    # 可视化模型训练过程
    plt.rcParams['font.sans-serif'] = ['SF Mono']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 360  # 图片像素
    plt.rcParams['figure.dpi'] = 360  # 分辨率
    
    plt.figure(figsize = (12, 4))
    # 损失函数
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.loss, "ro-", label = "Train loss")
    plt.plot(train_process.epoch, train_process.val_loss, "ro-", label = "Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    # 精度
    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.acc, "ro-", label = "Train acc")
    plt.plot(train_process.epoch, train_process.val_acc, "bs-", label = "Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


# prepare data
train_corpus = read_corpus(train_corpus_path)
dct = build_dict(train_corpus)

# build train dataloader
np.random.shuffle(train_corpus)
train_ds = NER_DataSet(train_corpus[:], dct)
train_dl = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True, drop_last = True, num_workers = 0)

# build validate dataset
val_corpus = read_corpus(val_corpus_path)
val_ds = NER_DataSet(val_corpus, dct)
val_dl = DataLoader(val_ds, batch_size = BATCH_SIZE)

# build test dataset
test_corpus = read_corpus(test_corpus_path)
test_ds = NER_DataSet(test_corpus, dct)
test_dl = DataLoader(test_ds, batch_size = BATCH_SIZE)

# create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##
model = BiLSTM_CRF(Config())
saved_model_name = "yidu_bilstm_crf.pth"
# load saved model
if os.path.isfile(path.weights_path + saved_model_name):
    print('load weight from ' + path.weights_path + saved_model_name)
    model.load_state_dict(torch.load(path.weights_path + saved_model_name, map_location = torch.device(device)))

model = model.to(device)
lr = 0.001  # 加载模型重新训练需要调整到上次的学习率
# kernel_regularizer = 0
optimizer = optim.Adam(model.parameters(), lr = lr)
# optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = kernel_regularizer)

##
# summary
sequence, tags = next(iter(train_dl))
sequence_cuda = sequence.to(device)
tags_cuda = tags.to(device)
mask_cuda = sequence_cuda > 0
pms.summary(model, sequence_cuda.long(), tags_cuda.long(), mask_cuda.long(),
            show_input = True, print_summary = True)

##
# train model
history = train(model, optimizer, train_dl, val_dl, device = device, epochs = 100, reduce_lr_epochs = 3,
                early_stop_epochs = 10, weights_save_path = saved_model_name, save_each_epoch = True, clip_grad = 1.0)

plot(history)

##
# evaluate
metric_train = evaluate(model, train_dl, device, toast = "predicting train set", verbose = True)
print("\nTrain set")
print(metric_train.report())

metric_val = evaluate(model, val_dl, device, toast = "predicting validate set", verbose = True)
print("\nValidate set")
print(metric_val.report())

metric_test = evaluate(model, test_dl, device, toast = "predicting test set", verbose = True)
print("\nTest set")
print(metric_test.report())
