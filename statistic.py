from collections import Counter

with open('./datasets/yidu_test.txt', encoding = 'utf-8') as f:
    # count = 0
    # max = 0
    # index = 0
    # chars = []
    # for i in f.readlines():
    #     index += 1
    #     chars.append(i.strip('\n').split('\t')[0])
    #     if i.strip('\n') == '':
    #         if count > max:
    #             max = count
    #             if count == 108:
    #                 print(index)
    #         count = 0
    #     else:
    #         count += 1
    # print(max)
    # print(len(set(chars)))
    
    lines = f.readlines()
    count = 0
    sentense_len = []
    for i in lines:
        count += 1
        if i.strip('\n') == '':
            sentense_len.append(count)
            count = 0
    print(max(sentense_len))
    print(len(sentense_len))
    freq = dict(Counter(sentense_len))
    print(sorted(freq.items(), key = lambda d: d[0], reverse = True))
    print(sorted(freq.items(), key = lambda d: d[1], reverse = True))
    
    count_large = 0
    for length in sentense_len:
        if length > 256:
            count_large += 1
    print(count_large)
    print(count_large / len(sentense_len))
    
    max_len = 256
    count_total = 0
    for len, f in sorted(freq.items(), key = lambda d: d[0], reverse = True):
        if len > max_len:
            count_total += 1
    print(count_total)
