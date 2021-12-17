import torch
from collections import defaultdict

import path
from config import Config, dct
from data import sentence_to_tensor, BEGIN_TAGS, OUT_TAG, get_entity_type
from model import BiLSTM_CRF


def predict_sentence_tags(model, sentence, dct, device = None):
    sequence = sentence_to_tensor(sentence, dct)
    sequence = sequence.unsqueeze(0)
    with torch.no_grad():
        sequence_cuda = sequence.to(device)
        mask_cuda = sequence_cuda > 0
        
        tags_pred = model.predict(sequence_cuda, mask_cuda)
    
    return tags_pred[0]


def get_entity(sentence, tags):
    entity_dict = defaultdict(set)
    
    entity_start_index = -1
    entity = None
    entity_type = None
    for index, tag in enumerate(tags):
        entity = None
        if tag in BEGIN_TAGS:
            if entity_start_index == -1:
                entity_type = get_entity_type(tag)
                entity_start_index = index
            else:
                entity = sentence[entity_start_index: index]
                entity_dict[entity_type].add(entity)
                
                entity_type = get_entity_type(tag)
                entity_start_index = index
        
        if tag == OUT_TAG:
            if entity_start_index != -1:
                entity = sentence[entity_start_index: index]
                entity_dict[entity_type].add(entity)
                entity_start_index = -1
    
    return dict(entity_dict)


if __name__ == '__main__':
    model = BiLSTM_CRF(Config())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(path.weights_path + path.saved_model_name, map_location = torch.device(device)))
    sentence = '右横隔见数枚肿大淋巴结较前退缩，现显示不清（4:9）。左肺下叶后基底段见不规则结节灶较前稍缩小，现最大截面约1.1*0.9mm（7.15），边界尚清；右肺中下叶见散在数枚直径小于0.5cm的模糊小结节影与前大致相仿（7:18、30、36）；双肺尖见少许斑片、条索影较前无明显变化，余肺野未见明显实质性病变。'
    tags = predict_sentence_tags(model, sentence, dct, device)
    print(get_entity(sentence, tags))
