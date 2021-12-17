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
3. 修改data.py中的文本最长长度SEQUENCE_MAX_LENGTH
4. 使用utils/load_data.py生成label2id.txt文件，将其中的内容填到data.py的TAG_MAP中。注意：序号必须从1开始
5. 修改data.py中的len_tag_dict，值等于TAG_MAP的最大值
6. 修改data.py中build_dict(corpus, num_words = 6000)的num_words，为词表长度，词表按词频生成，超过num_words的将被忽略
7. 根据需要修改model.py模型结构
8. 修改config.py的超参数
9. 训练前**debug看下main.py的train_dl,train_ds对不对**
10. 训练，注意，必须传入验证集数据，如果没有验证集，就把测试集传入

## Model

模型的结构大致如下，这里 BiLSTM 层的输入为字向量。Bi-LSTM 对每个字进行编码，然后经过 softmax 后，每个词对应一个长度为 `len(tags)` 的向量，在不使用 CRF 的方法中，就取这个向量中最大的值的位置作为预测的 tag 了，可能会不符合BIO标签的规则。

这里每个词的对应的向量作为 CRF 的输入，CRF 会最大化整个序列的概率，学习BIO的规则，保证输出格式是合法的。

在 PyTorch 中没有 CRF 层，这里使用了 [AllenNLP](https://github.com/allenai/allennlp) 中的 CRF 实现


## Config

在条件随机场中存在一个状态转移矩阵，在这里此状态转移矩阵就包含的是不同 tag 之间转移的概率。但并不是任何状态之间都能进行转移的，比如 `B-PER` 就不可能转移到 `I-LOC` 上。`condtraints` 就用来指明那些状态之间可以转移，这样将极大地减少可能性，在训练和解码过程中，能够大幅提升速度。请务必指定此参数，其创建方法见 `data.py`。

## Train

```
runfile('/Volumes/Riesling/TRAIN/Torch-base/src/BiLSTM_CRF/main.py', wdir='/Volumes/Riesling/TRAIN/Torch-base/src/BiLSTM_CRF')
-----------------------------------------------------------------------------------------------------------
               Layer (type)                                    Input Shape         Param #     Tr. Param #
===========================================================================================================
                Embedding-1                                      [100, 32]         180,300         180,300
                     LSTM-2     [100, 32, 300], [2, 32, 256], [2, 32, 256]       1,142,784       1,142,784
                     LSTM-3     [100, 32, 512], [2, 32, 128], [2, 32, 128]         657,408         657,408
                  Dropout-4                                 [100, 32, 256]               0               0
                   Linear-5                                 [32, 100, 256]           7,453           7,453
   ConditionalRandomField-6            [32, 100, 29], [32, 100], [32, 100]           1,860             899
===========================================================================================================
Total params: 1,989,805
Trainable params: 1,988,844
Non-trainable params: 961
-----------------------------------------------------------------------------------------------------------
training on  cpu
epoch [1]: 100%|██████████| 29/29 [00:26<00:00,  1.09it/s, loss=42.7]
2021-12-17 14:52:37,207 - epoch 1 - loss: 42.6906 acc: 0.2409 - test_acc: 0.1804
epoch [2]: 100%|██████████| 29/29 [00:28<00:00,  1.02it/s, loss=17.6]
2021-12-17 14:53:19,099 - epoch 2 - loss: 17.5649 acc: 0.6874 - test_acc: 0.6554
epoch [3]: 100%|██████████| 29/29 [00:32<00:00,  1.10s/it, loss=10.7]
2021-12-17 14:54:01,811 - epoch 3 - loss: 10.6861 acc: 0.7702 - test_acc: 0.6908
epoch [4]: 100%|██████████| 29/29 [00:31<00:00,  1.07s/it, loss=7.76]
2021-12-17 14:54:43,429 - epoch 4 - loss: 7.7551 acc: 0.8336 - test_acc: 0.7633
epoch [5]: 100%|██████████| 29/29 [00:29<00:00,  1.03s/it, loss=5.89]
2021-12-17 14:55:23,635 - epoch 5 - loss: 5.8919 acc: 0.8689 - test_acc: 0.7907
epoch [6]: 100%|██████████| 29/29 [00:29<00:00,  1.00s/it, loss=4.68]
2021-12-17 14:56:01,725 - epoch 6 - loss: 4.6774 acc: 0.8798 - test_acc: 0.7858
epoch [7]: 100%|██████████| 29/29 [00:38<00:00,  1.32s/it, loss=4.03]
2021-12-17 14:56:53,304 - epoch 7 - loss: 4.0329 acc: 0.9052 - test_acc: 0.7858
epoch [8]: 100%|██████████| 29/29 [00:35<00:00,  1.21s/it, loss=3.34]
2021-12-17 14:57:41,694 - epoch 8 - loss: 3.3428 acc: 0.9118 - test_acc: 0.8180
epoch [9]: 100%|██████████| 29/29 [00:38<00:00,  1.32s/it, loss=2.98]
2021-12-17 14:58:29,565 - epoch 9 - loss: 2.9814 acc: 0.9217 - test_acc: 0.7762
epoch [10]: 100%|██████████| 29/29 [00:36<00:00,  1.26s/it, loss=2.53]
2021-12-17 14:59:15,809 - epoch 10 - loss: 2.5263 acc: 0.9298 - test_acc: 0.7971
```

## Evaluate

```python
metric_test = evaluate(model, test_dl, device, verbose = True)
print(metric_test.report())
```

测试集上的表现：

```
predicting training set: 100%|██████████| 29/29 [00:08<00:00,  3.46it/s]
Train set
            ANATOMY     SIGN        QUANTITY    ORGAN       TEXTURE     DISEASE     DENSITY     BOUNDARY    MARGIN      DIAMETER    SHAPE       TREATMENT   LUNGFIELD   NATURE      
precision   0.92        0.93        0.97        0.86        0.93        0.95        0.90        1.00        1.00        0.95        0.88        0.91        1.00        1.00        
recall      0.92        0.93        0.94        0.79        0.88        0.93        0.90        1.00        1.00        0.95        0.77        0.89        0.92        1.00        
f1          0.92        0.93        0.95        0.82        0.90        0.94        0.90        1.00        1.00        0.95        0.82        0.90        0.96        1.00        
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
precision   0.93
recall      0.92
f1          0.93

predicting test set: 100%|██████████| 5/5 [00:01<00:00,  3.85it/s]
Test set
            ANATOMY     SIGN        QUANTITY    ORGAN       TEXTURE     DISEASE     DENSITY     BOUNDARY    MARGIN      DIAMETER    SHAPE       TREATMENT   LUNGFIELD   NATURE      
precision   0.83        0.72        0.87        0.69        0.86        0.72        1.00        0.75        0.83        1.00        0.71        0.56        0.83        1.00        
recall      0.81        0.76        0.85        0.64        1.00        0.61        1.00        1.00        0.83        1.00        0.83        0.38        1.00        0.86        
f1          0.82        0.74        0.86        0.67        0.92        0.66        1.00        0.86        0.83        1.00        0.77        0.45        0.91        0.92        
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
precision   0.79
recall      0.78
f1          0.79
```

## Predict

```python
model = BiLSTM_CRF(Config())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(path.weights_path + path.saved_model_name, map_location = torch.device(device)))
sentence = '右横隔见数枚肿大淋巴结较前退缩，现显示不清（4:9）。左肺下叶后基底段见不规则结节灶较前稍缩小，现最大截面约1.1cm*0.9cm（7.15），边界尚清；右肺中下叶见散在数枚直径小于0.5cm的模糊小结节影与前大致相仿（7:18、30、36）；双肺尖见少许斑片、条索影较前无明显变化，余肺野未见明显实质性病变。'
tags = predict_sentence_tags(model, sentence, dct, device)
print(get_entity(sentence, tags))
```
预测结果：
```
{
    'ANATOMY': {'9mm', '右肺中下叶', '左肺下叶后基底段', '右横隔', '双肺尖'}, 
    'SIGN': {'明显实质性病变', '肿大淋巴结较前退缩', '斑片、条索影较前无明显变化', '较前稍缩小', '不规则结节灶', '数枚', '小结节影'}, 
    'DIAMETER': {'1.1*'}, 
    'BOUNDARY': {'尚清'}, 
    'QUANTITY': {'少许', '未见'}
}
```







