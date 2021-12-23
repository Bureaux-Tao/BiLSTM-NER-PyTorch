import json
import random
from collections import Counter

len_list = []
split_char = '\t'
final_list = []
with open('datasets/subtask1_training_part1.jsonl', 'r', encoding = 'utf-8-sig') as f1:
    count1 = 0
    lines = f1.readlines()
    items_1 = []
    for line in lines:
        if line == '\n':
            continue
        items_1.append(json.loads(line.strip('\n')))
    
    for item in items_1:
        sentense_list = []
        len_list.append(len(item['originalText']))
        count1 += len(item['originalText'])
        for i_txt in item['originalText']:
            sentense_list.append({'w': i_txt, 't': 'O'})
        
        for i_entities in item['entities']:
            label = ""
            if i_entities['label_type'] == '疾病和诊断':
                label = "DISEASE"
            elif i_entities['label_type'] == '解剖部位':
                label = "ANATOMY"
            elif i_entities['label_type'] == '手术':
                label = "OPERATION"
            elif i_entities['label_type'] == '药物':
                label = "DRUG"
            elif i_entities['label_type'] == '影像检查':
                label = "TESTIMAGE"
            elif i_entities['label_type'] == '实验室检验':
                label = "TESTLAB"
            sentense_list[i_entities['start_pos']]['t'] = "B-" + label
            for i_index in range(i_entities['start_pos'] + 1, i_entities['end_pos']):
                sentense_list[i_index]['t'] = 'I-' + label
        
        final_list.append(sentense_list)
    print(count1)

with open('datasets/subtask1_training_part2.jsonl', 'r', encoding = 'utf-8-sig') as f2:
    count2 = 0
    lines = f2.readlines()
    items_2 = []
    for line in lines:
        if line == '\n':
            continue
        items_2.append(json.loads(line.strip('\n')))
    
    for item in items_2:
        sentense_list = []
        len_list.append(len(item['originalText']))
        count2 += len(item['originalText'])
        for i_txt in item['originalText']:
            sentense_list.append({'w': i_txt, 't': 'O'})
        
        for i_entities in item['entities']:
            label = ""
            if i_entities['label_type'] == '疾病和诊断':
                label = "DISEASE"
            elif i_entities['label_type'] == '解剖部位':
                label = "ANATOMY"
            elif i_entities['label_type'] == '手术':
                label = "OPERATION"
            elif i_entities['label_type'] == '药物':
                label = "DRUG"
            elif i_entities['label_type'] == '影像检查':
                label = "TESTIMAGE"
            elif i_entities['label_type'] == '实验室检验':
                label = "TESTLAB"
            sentense_list[i_entities['start_pos']]['t'] = "B-" + label
            for i_index in range(i_entities['start_pos'] + 1, i_entities['end_pos']):
                sentense_list[i_index]['t'] = 'I-' + label
        
        final_list.append(sentense_list)
    print(count2)

random.shuffle(final_list)
train = final_list[:int(len(final_list) * 0.8)]
val = final_list[int(len(final_list) * 0.8):int(len(final_list) * 0.9)]
test = final_list[int(len(final_list) * 0.9):]
with open('datasets/yidu_train.txt', 'a', encoding = 'utf-8-sig') as f11:
    with open('datasets/yidu_validate.txt', 'a', encoding = 'utf-8-sig') as f22:
        with open('datasets/yidu_test.txt', 'a', encoding = 'utf-8-sig') as f33:
            for i_final_list in train:
                count = 0
                for j_final_list in i_final_list:
                    if j_final_list['w'] == '。' and count > 20:
                        f11.write(j_final_list['w'] + split_char + j_final_list['t'] + '\n\n')
                        count = 0
                    else:
                        f11.write(j_final_list['w'] + split_char + j_final_list['t'] + '\n')
                        count += 1
                        
            for i_final_list in val:
                count = 0
                for j_final_list in i_final_list:
                    if j_final_list['w'] == '。' and count > 20:
                        f22.write(j_final_list['w'] + split_char + j_final_list['t'] + '\n\n')
                        count = 0
                    else:
                        f22.write(j_final_list['w'] + split_char + j_final_list['t'] + '\n')
                        count += 1
                        
            for i_final_list in test:
                count = 0
                for j_final_list in i_final_list:
                    if j_final_list['w'] == '。' and count > 20:
                        f33.write(j_final_list['w'] + split_char + j_final_list['t'] + '\n\n')
                        count = 0
                    else:
                        f33.write(j_final_list['w'] + split_char + j_final_list['t'] + '\n')
                        count += 1

# with open('datasets/subtask1_test_set_with_answer.jsonl', 'r', encoding = 'utf-8-sig') as f3:
#     with open('datasets/yidu_test.txt', 'a', encoding = 'utf-8-sig') as f33:
#         count3 = 0
#         final_list = []
#         lines = f3.readlines()
#         items_3 = []
#         for line in lines:
#             if line == '\n':
#                 continue
#             items_3.append(json.loads(line.strip('\n')))
#
#         for item in items_3:
#             sentense_list = []
#             len_list.append(len(item['originalText']))
#             count3 += len(item['originalText'])
#             for i_txt in item['originalText']:
#                 sentense_list.append({'w': i_txt, 't': 'O'})
#
#             for i_entities in item['entities']:
#                 label = ""
#                 if i_entities['label_type'] == '疾病和诊断':
#                     label = "DISEASE"
#                 elif i_entities['label_type'] == '解剖部位':
#                     label = "ANATOMY"
#                 elif i_entities['label_type'] == '手术':
#                     label = "OPERATION"
#                 elif i_entities['label_type'] == '药物':
#                     label = "DRUG"
#                 elif i_entities['label_type'] == '影像检查':
#                     label = "TESTIMAGE"
#                 elif i_entities['label_type'] == '实验室检验':
#                     label = "TESTLAB"
#                 sentense_list[i_entities['start_pos']]['t'] = "B-" + label
#                 for i_index in range(i_entities['start_pos'] + 1, i_entities['end_pos']):
#                     sentense_list[i_index]['t'] = 'I-' + label
#
#             final_list.append(sentense_list)
#         print(count3)
#         for i_final_list in final_list:
#             count = 0
#             for j_final_list in i_final_list:
#                 if j_final_list['w'] == '。' and count > 20:
#                     f33.write(j_final_list['w'] + split_char + j_final_list['t'] + '\n\n')
#                     count = 0
#                 else:
#                     f33.write(j_final_list['w'] + split_char + j_final_list['t'] + '\n')
#                     count += 1

freq = dict(Counter(len_list))
for i in sorted(freq.items(), key = lambda d: d[0], reverse = True):
    print(i)
