import os

event_type = "yidu_4k"

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前地址
proj_path = current_dir

weights_path = proj_path + "/weights/"

train_file_path = proj_path + "/datasets/yidu_train.txt"
test_file_path = proj_path + "/datasets/yidu_test.txt"
val_file_path = proj_path + "/datasets/yidu_validate.txt"

vocab_path = proj_path + "/datasets/dct.pkl"
