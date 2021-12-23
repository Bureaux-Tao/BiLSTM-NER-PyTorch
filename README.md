## BiLSTM-CRF for Name Entity Recognition PyTorch version

A PyTorch implemention of Bi-LSTM-CRF model for Chinese Named Entity Recognition. 

使用 PyTorch 实现 Bi-LSTM-CRF 模型，用来完成中文命名实体识别任务。


## Dataset

Yidu-S4K数据集，对于给定的一组电子病历纯文本文档，任务的目标是识别并抽取出与医学临床相关的实体提及（entity mention），并将它们归类到预定义类别（pre-defined categories），比如疾病、治疗、检查检验等。

1. 疾病和诊断：医学上定义的疾病和医生在临床工作中对病因、病生理、分型分期等所作的判断。
2. 检查：	影像检查（X线、CT、MR、PETCT等）+造影+超声+心电图，未避免检查操作与手术操作过多冲突，不包含此外其它的诊断性操作，如胃镜、肠镜等。
3. 检验：	在实验室进行的物理或化学检查，本期特指临床工作中检验科进行的化验，不含免疫组化等广义实验室检查
4. 手术：	医生在患者身体局部进行的切除、缝合等治疗，是外科的主要治疗方法。
5. 药物：	用于疾病治疗的具体化学物质。
6. 解剖部位：	指疾病、症状和体征发生的人体解剖学部位。

任务一数据结构：
任务一数据每一行为一个json
json key 为['originalText','entities'] 即原文和实体列表
json["entities"]为列表，每个元素代表一个实体entity，其中有该实体在原文中的起始位置start_pos,结束位置end_pos,以及实体类型

训练样本1000条，提交的测试样本379条，经过transfer_yidu.py处理后转成BIO格式，形如：

```
心	B-TESTIMAGE
脏	I-TESTIMAGE
彩	I-TESTIMAGE
超	I-TESTIMAGE
：	O
右	B-ANATOMY
房	I-ANATOMY
、	O
右	B-ANATOMY
室	I-ANATOMY
稍	O
增	O
大	O
，	O
E	B-TESTLAB
F	I-TESTLAB
正	O
常	O
。	O
```

ATTENTION:
- 字与标签之间用tab（"\t"）隔开
- 其中句子与句子之间使用空行隔开
- 文件最后以两个换行结束

训练数据和测试数据存放在 `datasets` 目录下，在 `data.py` 中有具体读取数据的代码。

## Project Structure

```
./
├── README.md
├── __pycache__
├── config.py                       模型超参数配置
├── data.py                         数据转换 加载词表
├── datasets                        数据集
│   ├── dct.pkl                     词表
│   ├── subtask1_test_set_with_answer.jsonl
│   ├── subtask1_training_part1.jsonl
│   ├── subtask1_training_part2.jsonl
│   ├── yidu_submit.txt             提交最终评测的379条样本
│   ├── yidu_test.txt               自己划分的测试集
│   ├── yidu_train.txt              验证集
│   └── yidu_validate.txt           训练集
├── evaluate.py                     评估模型
├── log                             nohup训练输出日志
│   └── nohup.out
├── main.py                         训练
├── metric.py                       计算f1评估指标
├── model.py                        模型构建
├── path.py                         所有路径
├── predict.py                      实体抽取预测
├── yidu_4k_label2id.json           所有label
├── requirements.txt                pip包
├── statistic.py                    统计句子长度的工具
├── trainer.py                      训练器构建
├── transfer_yidu.py                转换原格式到BIO格式
├── utils
│   ├── __init__.py
│   ├── __pycache__
│   ├── earlystopping.py            早停工具
│   └── load_data.py                构建label2id工具
└── weights                         权重
    └── xxx.pth

6 directories, 32 files
```

## Steps

1. 替换数据集
2. 修改path.py中的地址
3. 删掉datasets/dct.pkl旧词表
4. 修改data.py中的文本最长长度SEQUENCE_MAX_LENGTH
5. 使用utils/load_data.py生成label2id.txt文件，将其中的内容填到data.py的TAG_MAP中。注意：序号必须从0开始
6. 修改data.py中的len_tag_dict，值等于TAG_MAP的最大值
7. 修改data.py中build_dict(corpus, num_words = 1800)的num_words，为词表长度，词表按词频生成，超过num_words的将被忽略
8. 根据需要修改model.py模型结构
9. 修改config.py的超参数
10. 训练前**debug看下main.py的train_dl,train_ds对不对**
11. 训练，注意，train()中的验证集参数val_dl必须有数据传入，如果没划验证集，可以把测试集传入

## Model

模型的结构大致如下，这里 BiLSTM 层的输入为字向量。Bi-LSTM 对每个字进行编码，然后经过 softmax 后，每个词对应一个长度为 `len(tags)` 的向量，在不使用 CRF 的方法中，就取这个向量中最大的值的位置作为预测的 tag 了，可能会不符合BIO标签的规则。

这里每个词的对应的向量作为 CRF 的输入，CRF 会最大化整个序列的概率，学习BIO的规则，保证输出格式是合法的。

在 PyTorch 中没有 CRF 层，这里使用了 [AllenNLP](https://github.com/allenai/allennlp) 中的 CRF 实现


## Config

在条件随机场中存在一个状态转移矩阵，在这里此状态转移矩阵就包含的是不同 tag 之间转移的概率。但并不是任何状态之间都能进行转移的，比如 `B-PER` 就不可能转移到 `I-LOC` 上。`condtraints` 就用来指明那些状态之间可以转移，这样将极大地减少可能性，在训练和解码过程中，能够大幅提升速度。请务必指定此参数，其创建方法见 `data.py`。

## Train

```
--------------------------------------------------------------------------------------------------------------
               Layer (type)                                       Input Shape         Param #     Tr. Param #
==============================================================================================================
                Embedding-1                                        [256, 128]         536,400         536,400
                     LSTM-2     [256, 128, 300], [2, 128, 256], [2, 128, 256]       1,142,784       1,142,784
                  Dropout-3                                   [256, 128, 512]               0               0
                     LSTM-4     [256, 128, 512], [2, 128, 128], [2, 128, 128]         657,408         657,408
                  Dropout-5                                   [256, 128, 256]               0               0
          TimeDistributed-6                                   [128, 256, 256]           3,341           3,341
                  Dropout-7                                    [128, 256, 13]               0               0
   ConditionalRandomField-8            [128, 256, 13], [128, 256], [128, 256]             420             195
==============================================================================================================
Total params: 2,340,353
Trainable params: 2,340,128
Non-trainable params: 225
--------------------------------------------------------------------------------------------------------------
training on  cpu
epoch [1]: 100%|█████████████████████| 38/38 [02:16<00:00,  3.59s/it, loss=65.4]
2021-12-23 11:35:02,965 - epoch 1 - loss: 65.3639 acc: 0.2243 val_loss: 21.6087 val_acc: 0.1964
epoch [2]: 100%|█████████████████████| 38/38 [02:04<00:00,  3.29s/it, loss=29.7]
2021-12-23 11:38:10,371 - epoch 2 - loss: 29.7486 acc: 0.5719 val_loss: 12.0026 val_acc: 0.5058
epoch [3]: 100%|█████████████████████| 38/38 [02:06<00:00,  3.32s/it, loss=22.1]
2021-12-23 11:41:14,555 - epoch 3 - loss: 22.0580 acc: 0.7099 val_loss: 9.2177 val_acc: 0.6453
epoch [4]: 100%|█████████████████████| 38/38 [02:06<00:00,  3.33s/it, loss=18.2]
2021-12-23 11:44:18,399 - epoch 4 - loss: 18.1640 acc: 0.7833 val_loss: 8.2971 val_acc: 0.6895
epoch [5]: 100%|█████████████████████| 38/38 [02:02<00:00,  3.24s/it, loss=16.1]
2021-12-23 11:47:23,206 - epoch 5 - loss: 16.1152 acc: 0.8306 val_loss: 7.2306 val_acc: 0.7330
epoch [6]: 100%|█████████████████████| 38/38 [01:58<00:00,  3.11s/it, loss=14.1]
2021-12-23 11:50:24,777 - epoch 6 - loss: 14.0959 acc: 0.8500 val_loss: 7.1640 val_acc: 0.7419
epoch [7]: 100%|█████████████████████| 38/38 [02:05<00:00,  3.30s/it, loss=12.5]
2021-12-23 11:53:32,239 - epoch 7 - loss: 12.5184 acc: 0.8639 val_loss: 7.6330 val_acc: 0.7463
EarlyStopping counter: 1 out of 10
epoch [8]: 100%|█████████████████████| 38/38 [02:07<00:00,  3.34s/it, loss=11.1]
2021-12-23 11:56:41,437 - epoch 8 - loss: 11.0887 acc: 0.8795 val_loss: 7.3306 val_acc: 0.7479
EarlyStopping counter: 2 out of 10
epoch [9]: 100%|███████████████████████| 38/38 [02:01<00:00,  3.20s/it, loss=10]
2021-12-23 11:59:43,054 - epoch 9 - loss: 9.9951 acc: 0.8972 val_loss: 7.5721 val_acc: 0.7584
EarlyStopping counter: 3 out of 10
epoch [10]: 100%|████████████████████| 38/38 [02:01<00:00,  3.20s/it, loss=8.87]
2021-12-23 12:02:47,965 - epoch 10 - loss: 8.8729 acc: 0.9147 val_loss: 7.6724 val_acc: 0.7551
EarlyStopping counter: 4 out of 10
Epoch    10: reducing learning rate of group 0 to 5.0000e-04.
epoch [11]: 100%|█████████████████████| 38/38 [02:02<00:00,  3.23s/it, loss=7.8]
2021-12-23 12:05:51,141 - epoch 11 - loss: 7.8048 acc: 0.9317 val_loss: 8.5222 val_acc: 0.7474
EarlyStopping counter: 5 out of 10
epoch [12]: 100%|████████████████████| 38/38 [02:06<00:00,  3.32s/it, loss=7.14]
2021-12-23 12:08:59,069 - epoch 12 - loss: 7.1434 acc: 0.9410 val_loss: 9.1695 val_acc: 0.7546
EarlyStopping counter: 6 out of 10
epoch [13]: 100%|████████████████████| 38/38 [02:09<00:00,  3.40s/it, loss=6.76]
2021-12-23 12:12:10,392 - epoch 13 - loss: 6.7571 acc: 0.9526 val_loss: 9.0149 val_acc: 0.7623
EarlyStopping counter: 7 out of 10
epoch [14]: 100%|████████████████████| 38/38 [01:56<00:00,  3.06s/it, loss=6.23]
2021-12-23 12:15:09,792 - epoch 14 - loss: 6.2310 acc: 0.9552 val_loss: 9.7749 val_acc: 0.7468
EarlyStopping counter: 8 out of 10
Epoch    14: reducing learning rate of group 0 to 2.5000e-04.
epoch [15]: 100%|████████████████████| 38/38 [01:59<00:00,  3.14s/it, loss=5.84]
2021-12-23 12:18:09,680 - epoch 15 - loss: 5.8399 acc: 0.9615 val_loss: 10.0699 val_acc: 0.7496
EarlyStopping counter: 9 out of 10
epoch [16]: 100%|████████████████████| 38/38 [02:04<00:00,  3.28s/it, loss=5.59]
2021-12-23 12:21:18,497 - epoch 16 - loss: 5.5908 acc: 0.9668 val_loss: 10.4283 val_acc: 0.7590
EarlyStopping counter: 10 out of 10
Early stopping after epoch 16
predicting train set: 100%|█████████████████████| 38/38 [00:52<00:00,  1.39s/it]
Train set
            ANATOMY     TESTIMAGE   DISEASE     OPERATION   DRUG        TESTLAB
precision   0.959       0.979       0.978       0.954       0.984       0.948
recall      0.957       0.960       0.971       0.963       0.982       0.949
f1          0.958       0.969       0.975       0.959       0.983       0.949
------------------------------------------------------------------------------------
precision   0.966
recall      0.963
f1          0.965
predicting validate set: 100%|████████████████████| 6/6 [00:06<00:00,  1.10s/it]
Validate set
            ANATOMY     TESTIMAGE   DISEASE     OPERATION   DRUG        TESTLAB
precision   0.764       0.849       0.671       0.784       0.825       0.798
recall      0.742       0.730       0.671       0.770       0.932       0.766
f1          0.753       0.785       0.671       0.777       0.875       0.782
------------------------------------------------------------------------------------
precision   0.760
recall      0.751
f1          0.755
```

## Evaluate

```python
metric_test = evaluate(model, test_dl, device, verbose = True)
print(metric_test.report())
```

测试集上的表现：

```
            ANATOMY     TESTIMAGE   DISEASE     OPERATION   DRUG        TESTLAB     
precision   0.813       0.893       0.777       0.758       0.836       0.822       
recall      0.815       0.852       0.733       0.774       0.874       0.807       
f1          0.814       0.872       0.754       0.766       0.854       0.815       
------------------------------------------------------------------------------------
precision   0.810
recall      0.802
f1          0.806
predicting test set: 100%|██████████████████████| 16/16 [00:22<00:00,  1.43s/it]
```

最终提交的subtask1_test_set_with_answer.jsonl表现

```
            ANATOMY     TESTIMAGE   DISEASE     OPERATION   DRUG        TESTLAB     
precision   0.823       0.788       0.749       0.741       0.769       0.547       
recall      0.803       0.788       0.738       0.816       0.813       0.680       
f1          0.813       0.788       0.743       0.777       0.790       0.607       
------------------------------------------------------------------------------------
precision   0.771
recall      0.779
f1          0.775
```






