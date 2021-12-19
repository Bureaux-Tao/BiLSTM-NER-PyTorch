import os

event_type = "pulmonary"

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前地址
proj_path = current_dir

weights_path = proj_path + "/weights/"

train_file_path = proj_path + "/datasets/train_data.txt"
test_file_path = proj_path + "/datasets/test_data.txt"

vocab_path = proj_path + "/datasets/dct.pkl"
