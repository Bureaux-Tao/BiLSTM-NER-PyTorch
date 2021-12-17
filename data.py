import os
from collections import Counter
import pickle
import re

import torch
from torch.utils.data import Dataset, DataLoader
from allennlp.modules.conditional_random_field import allowed_transitions

import path

PAD_WORD = '<PAD>'
UNK_WORD = '<UNK>'
NUM_WORD = '<NUM>'
ENG_WORD = '<ENG>'

# 文档最大长度限制
SEQUENCE_MAX_LENGTH = 100

# tags, BIO
TAG_MAP = {
    "O": 0,
    "B-ANATOMY": 1,
    "I-ANATOMY": 2,
    "B-SIGN": 3,
    "I-SIGN": 4,
    "B-QUANTITY": 5,
    "I-QUANTITY": 6,
    "B-ORGAN": 7,
    "I-ORGAN": 8,
    "B-TEXTURE": 9,
    "I-TEXTURE": 10,
    "B-DISEASE": 11,
    "I-DISEASE": 12,
    "B-DENSITY": 13,
    "I-DENSITY": 14,
    "B-BOUNDARY": 15,
    "I-BOUNDARY": 16,
    "B-MARGIN": 17,
    "I-MARGIN": 18,
    "B-DIAMETER": 19,
    "I-DIAMETER": 20,
    "B-SHAPE": 21,
    "I-SHAPE": 22,
    "B-TREATMENT": 23,
    "I-TREATMENT": 24,
    "B-LUNGFIELD": 25,
    "I-LUNGFIELD": 26,
    "B-NATURE": 27,
    "I-NATURE": 28
}

len_tag_dict = 28
n = list(range(len_tag_dict + 1))
BEGIN_TAGS = set(n[1::2])
OUT_TAG = TAG_MAP['O']

TAG_MAP_REVERSED = {
    v: k for k, v in TAG_MAP.items()
}

condtraints = allowed_transitions('BIO', TAG_MAP_REVERSED)


def get_entity_type(tag):
    return TAG_MAP_REVERSED[tag].split('-')[1]


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: samples
    """
    corpus = []
    
    with open(corpus_path, encoding = 'utf-8', mode = 'r') as fin:
        sequence, tags = [], []
        for line in fin:
            if line != '\n':
                [char, tag] = line.strip().split()
                sequence.append(char)
                tags.append(tag)
            else:
                corpus.append((sequence, tags))
                sequence, tags = [], []
    
    return corpus


def build_dict(corpus, num_words = 6000):
    dct_file = path.vocab_path
    if os.path.exists(dct_file):
        with open(dct_file, mode = 'rb') as fin:
            dct = pickle.load(fin)
            return dct
    
    counter = Counter()
    for sequence, _ in corpus:
        counter.update(sequence)
    
    words = [w for w, c in counter.most_common(num_words - 4)]
    words = [PAD_WORD, UNK_WORD, NUM_WORD, ENG_WORD] + words
    
    dct = {word: i for i, word in enumerate(words)}
    
    with open(dct_file, mode = 'wb') as fout:
        pickle.dump(dct, fout)
    
    return dct


def sentence_to_tensor(sentence, dct):
    UNK = dct[UNK_WORD]
    idx = [dct.get(w, UNK) for w in sentence]
    idx = torch.tensor(idx, dtype = torch.long)
    return idx


class NER_DataSet(Dataset):
    def __init__(self, corpus, dictionary, sequence_max_length = SEQUENCE_MAX_LENGTH):
        self.sequence_max_length = sequence_max_length
        self.dct = dictionary
        self.samples = self.process_corpus(corpus)
    
    def __len__(self):
        return len(self.samples)
    
    # p[key] 取值，当实例对象做p[key] 运算时，会调用类中的方法__getitem__
    # 一般如果想使用索引访问元素时，就可以在类中定义这个方法（__getitem__(self, key) ）
    def __getitem__(self, i):
        return self.samples[i]
    
    def process_item(self, sequence, tags):
        UNK = self.dct[UNK_WORD]
        PAD = self.dct[PAD_WORD]
        UNM = self.dct[NUM_WORD]
        ENG = self.dct[ENG_WORD]
        
        if len(sequence) > self.sequence_max_length:  # 截断
            sequence = sequence[:self.sequence_max_length]
            tags = tags[:self.sequence_max_length]
        
        seq = sequence
        sequence = []
        for w in seq:  # 将文本转换为数值
            if w.isdigit():
                sequence.append(UNM)
            elif ('\u0041' <= w <= '\u005a') or ('\u0061' <= w <= '\u007a'):
                sequence.append(ENG)
            else:
                sequence.append(self.dct.get(w, UNK))  # 初值为UNK，赋值为取w的
        
        tags = [TAG_MAP[tag] for tag in tags]  # 将标签转换为数值
        
        if len(sequence) < self.sequence_max_length:  # 填充
            sequence += [PAD] * (self.sequence_max_length - len(sequence))
            tags += [0] * (self.sequence_max_length - len(tags))  # 填充0，不能和TAG_MAP值一样
        
        # 转成向量
        sequence = torch.tensor(sequence, dtype = torch.long)
        tags = torch.tensor(tags, dtype = torch.long)
        
        return sequence, tags
    
    def process_corpus(self, corpus):
        samples = [
            self.process_item(sequence, tags) for sequence, tags in corpus
            # 逐句取出处理
        ]
        return samples
