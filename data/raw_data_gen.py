import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="ICEWS14", type=str)
args = vars(parser.parse_args())  # 得到参数字典

dataset = args['dataset']
dataset_path = f'./{dataset}'

with open(os.path.join(dataset_path, 'entity2id.json')) as file:
    entity2id = json.load(file)
    id2entity = {v: k for k, v in entity2id.items()}
with open(os.path.join(dataset_path, 'relation2id.json')) as file:
    relation2id = json.load(file)
    id2relation = {v: k for k, v in relation2id.items()}
with open(os.path.join(dataset_path, 'ts2id.json')) as file:
    ts2id = json.load(file)
    id2ts = {v: k for k, v in ts2id.items()}

data_file_id = ['train.txt', 'valid.txt', 'test.txt']
data_file_raw = ['train_raw.txt', 'valid_raw.txt', 'test_raw.txt']
for id in range(len(data_file_id)):
    data_path_id = os.path.join(dataset_path, data_file_id[id])
    data_path_raw = os.path.join(dataset_path, data_file_raw[id])
    with open(data_path_id, 'r') as file:
        lines_id = file.readlines()
    with open(data_path_raw, 'w') as file:
        for line in lines_id:
            line_list = line.strip().split('\t')[:4]
            s, p, o, t = map(int, line_list)
            raw_line = f'{id2entity[s]}\t{id2relation[p]}\t{id2entity[o]}\t{id2ts[t]}\n'
            file.write(raw_line)
            

