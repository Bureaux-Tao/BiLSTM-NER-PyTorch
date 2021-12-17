with open('./datasets/test_data.txt', encoding = 'utf-8') as f:
    count = 0
    max = 0
    index = 0
    for i in f.readlines():
        index += 1
        if i.strip('\n') == '':
            if count > max:
                max = count
                if count == 108:
                    print(index)
            count = 0
        else:
            count += 1
    print(max)
