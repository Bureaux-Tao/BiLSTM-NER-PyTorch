## BiLSTM-CRF for Name Entity Recognition PyTorch version

A PyTorch implemention of Bi-LSTM-CRF model for Chinese Named Entity Recognition. 

使用 PyTorch 实现 Bi-LSTM-CRF 模型，用来完成中文命名实体识别任务。


## Dataset

三甲医院肺结节数据集，20000+字，BIO格式，形如：

```
中	B-ORG
共	I-ORG
中	I-ORG
央	I-ORG
致	O
中	B-ORG
国	I-ORG
致	I-ORG
公	I-ORG
党	I-ORG
十	I-ORG
一	I-ORG
大	I-ORG
的	O
贺	O
词	O
```
ATTENTION: 在处理自己数据集的时候需要注意：
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
│   ├── test_data.txt               训练集
│   └── train_data.txt              测试集
├── evaluate.py                     评估模型
├── log                             nohup训练输出日志
│   └── nohup.out
├── main.py                         训练
├── metric.py                       计算f1评估指标
├── model.py                        模型构建
├── path.py                         所有路径
├── predict.py                      实体抽取预测
├── pulmonary_label2id.json         所有label
├── requirements.txt                pip包
├── statistic.py                    统计最长句子长度
├── trainer.py                      训练器构建
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
5. 使用utils/load_data.py生成label2id.txt文件，将其中的内容填到data.py的TAG_MAP中。注意：序号必须从1开始
6. 修改data.py中的len_tag_dict，值等于TAG_MAP的最大值
7. 修改data.py中build_dict(corpus, num_words = 6000)的num_words，为词表长度，词表按词频生成，超过num_words的将被忽略
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
-----------------------------------------------------------------------------------------------------------
               Layer (type)                                    Input Shape         Param #     Tr. Param #
===========================================================================================================
                Embedding-1                                      [100, 32]         180,300         180,300
                     LSTM-2     [100, 32, 300], [2, 32, 128], [2, 32, 128]         440,320         440,320
                  Dropout-3                                 [100, 32, 256]               0               0
                     LSTM-4       [100, 32, 256], [2, 32, 64], [2, 32, 64]         164,864         164,864
                  Dropout-5                                 [100, 32, 128]               0               0
                   Linear-6                                 [32, 100, 128]           3,741           3,741
                  Dropout-7                                  [32, 100, 29]               0               0
   ConditionalRandomField-8            [32, 100, 29], [32, 100], [32, 100]           1,860             899
===========================================================================================================
Total params: 791,085
Trainable params: 790,124
Non-trainable params: 961
-----------------------------------------------------------------------------------------------------------
training on  cpu
epoch [1]: 100%|██████████| 29/29 [00:11<00:00,  2.42it/s, loss=55.4]
2021-12-19 15:17:05,769 - epoch 1 - loss: 55.3747 acc: 0.0947 - test_acc: 0.0853
epoch [2]: 100%|██████████| 29/29 [00:12<00:00,  2.36it/s, loss=32.8]
2021-12-19 15:17:22,125 - epoch 2 - loss: 32.8019 acc: 0.5530 - test_acc: 0.4815
epoch [3]: 100%|██████████| 29/29 [00:12<00:00,  2.26it/s, loss=23.4]
2021-12-19 15:17:40,381 - epoch 3 - loss: 23.3587 acc: 0.7065 - test_acc: 0.6425
epoch [4]: 100%|██████████| 29/29 [00:13<00:00,  2.11it/s, loss=19.5]
2021-12-19 15:17:59,396 - epoch 4 - loss: 19.4523 acc: 0.7680 - test_acc: 0.6763
epoch [5]: 100%|██████████| 29/29 [00:18<00:00,  1.57it/s, loss=16.6]
2021-12-19 15:18:22,579 - epoch 5 - loss: 16.6216 acc: 0.8021 - test_acc: 0.7246
epoch [6]: 100%|██████████| 29/29 [00:13<00:00,  2.12it/s, loss=15]
2021-12-19 15:18:40,516 - epoch 6 - loss: 15.0233 acc: 0.8359 - test_acc: 0.7536
epoch [7]: 100%|██████████| 29/29 [00:12<00:00,  2.29it/s, loss=13.4]
2021-12-19 15:18:57,356 - epoch 7 - loss: 13.4273 acc: 0.8657 - test_acc: 0.7778
epoch [8]: 100%|██████████| 29/29 [00:12<00:00,  2.37it/s, loss=12.2]
2021-12-19 15:19:13,600 - epoch 8 - loss: 12.1736 acc: 0.8841 - test_acc: 0.7729
epoch [9]: 100%|██████████| 29/29 [00:12<00:00,  2.36it/s, loss=11.3]
2021-12-19 15:19:29,989 - epoch 9 - loss: 11.3147 acc: 0.8988 - test_acc: 0.7907
epoch [10]: 100%|██████████| 29/29 [00:12<00:00,  2.36it/s, loss=10.4]
2021-12-19 15:19:46,385 - epoch 10 - loss: 10.3500 acc: 0.9053 - test_acc: 0.7681
epoch [11]: 100%|██████████| 29/29 [00:12<00:00,  2.36it/s, loss=9.53]
2021-12-19 15:20:02,812 - epoch 11 - loss: 9.5281 acc: 0.9168 - test_acc: 0.7907
epoch [12]: 100%|██████████| 29/29 [00:12<00:00,  2.35it/s, loss=9.04]
2021-12-19 15:20:19,287 - epoch 12 - loss: 9.0429 acc: 0.9258 - test_acc: 0.7794
epoch [13]: 100%|██████████| 29/29 [00:13<00:00,  2.11it/s, loss=8.44]
2021-12-19 15:20:37,410 - epoch 13 - loss: 8.4381 acc: 0.9340 - test_acc: 0.7681
epoch [14]: 100%|██████████| 29/29 [00:13<00:00,  2.15it/s, loss=7.79]
2021-12-19 15:20:55,747 - epoch 14 - loss: 7.7892 acc: 0.9353 - test_acc: 0.7842
epoch [15]: 100%|██████████| 29/29 [00:13<00:00,  2.14it/s, loss=7.63]
2021-12-19 15:21:13,580 - epoch 15 - loss: 7.6291 acc: 0.9423 - test_acc: 0.7778
epoch [16]: 100%|██████████| 29/29 [00:13<00:00,  2.15it/s, loss=7.23]
2021-12-19 15:21:31,319 - epoch 16 - loss: 7.2328 acc: 0.9480 - test_acc: 0.7826
epoch [17]: 100%|██████████| 29/29 [00:12<00:00,  2.36it/s, loss=6.83]
2021-12-19 15:21:47,606 - epoch 17 - loss: 6.8298 acc: 0.9550 - test_acc: 0.7746
epoch [18]: 100%|██████████| 29/29 [00:12<00:00,  2.35it/s, loss=6.37]
2021-12-19 15:22:04,803 - epoch 18 - loss: 6.3710 acc: 0.9522 - test_acc: 0.7746
epoch [19]: 100%|██████████| 29/29 [00:12<00:00,  2.30it/s, loss=6.05]
2021-12-19 15:22:21,579 - epoch 19 - loss: 6.0456 acc: 0.9605 - test_acc: 0.7810
epoch [20]: 100%|██████████| 29/29 [00:12<00:00,  2.32it/s, loss=5.68]
2021-12-19 15:22:38,388 - epoch 20 - loss: 5.6803 acc: 0.9570 - test_acc: 0.7633
```

## Evaluate

```python
metric_test = evaluate(model, test_dl, device, verbose = True)
print(metric_test.report())
```

测试集上的表现：

```
predicting training set: 100%|██████████| 29/29 [00:04<00:00,  7.18it/s]
Train set
            ANATOMY     SIGN        QUANTITY    ORGAN       TEXTURE     DISEASE     DENSITY     BOUNDARY    MARGIN      DIAMETER    SHAPE       TREATMENT   LUNGFIELD   NATURE      
precision   0.896       0.911       0.951       0.701       0.873       0.873       0.839       0.969       0.889       0.919       0.538       0.853       1.000       0.882       
recall      0.878       0.898       0.904       0.758       0.762       0.843       0.929       1.000       0.889       0.919       0.667       0.853       0.917       1.000       
f1          0.887       0.905       0.927       0.729       0.814       0.858       0.881       0.984       0.889       0.919       0.596       0.853       0.957       0.938       
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
precision   0.897
recall      0.882
f1          0.889
predicting test set: 100%|██████████| 5/5 [00:00<00:00,  7.94it/s]

Test set
            ANATOMY     SIGN        QUANTITY    ORGAN       TEXTURE     DISEASE     DENSITY     BOUNDARY    MARGIN      DIAMETER    SHAPE       TREATMENT   LUNGFIELD   NATURE      
precision   0.832       0.751       0.892       0.385       0.857       0.698       1.000       0.750       0.667       1.000       0.571       0.556       0.833       0.667       
recall      0.839       0.759       0.892       0.455       1.000       0.682       1.000       1.000       0.800       1.000       0.800       0.417       1.000       1.000       
f1          0.836       0.755       0.892       0.417       0.923       0.690       1.000       0.857       0.727       1.000       0.667       0.476       0.909       0.800       
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
precision   0.792
recall      0.807
f1          0.799
```

## Predict

```python
model = BiLSTM_CRF(Config())
saved_model_name = "xxx.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(path.weights_path + saved_model_name, map_location = torch.device(device)))
sentence = '右横隔见数枚肿大淋巴结较前退缩，现显示不清（4:9）。左肺下叶后基底段见不规则结节灶较前稍缩小，现最大截面约1.1*0.9mm（7.15），边界尚清；右肺中下叶见散在数枚直径小于0.5cm的模糊小结节影与前大致相仿（7:18、30、36）；双肺尖见少许斑片、条索影较前无明显变化，余肺野未见明显实质性病变。'
tags = predict_sentence_tags(model, sentence, dct, device)
print(get_entity(sentence, tags))
```
预测结果：
```
{
    'ANATOMY': {'右横隔', '右肺中下叶', '左肺下叶后基底段', '双肺尖'}, 
    'QUANTITY': {'未见', '数枚', '少许'}, 
    'SIGN': {'不清', '许斑片、条索影较前无明显变', '肿大淋巴结较前退缩', '不规则结节灶较前稍缩小', '明显实质性病变'}, 
    'DIAMETER': {'1.1*0.9mm'}
}
```







