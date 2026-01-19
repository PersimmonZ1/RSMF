from typing import Tuple, List
import os
import numpy as np
import json
import torch
from typing import Dict, Any
import math
import random


def get_stat(dataset_path: str) -> Tuple[int, int]:
    with open(os.path.join(dataset_path, 'stat.txt'), 'r') as file:
        line = file.readline()
        line = line.strip().split('\t')[:2]
        ent_num, rel_num = list(map(int, line))

    return ent_num, rel_num


def get_json_file(dataset_path, file_name):
    with open(os.path.join(dataset_path, file_name), 'r') as file:
        json_file = json.load(file)

    return json_file


def get_quad_list(dataset_path: str, data_file: str, rel_num: int):
    data_path = os.path.join(dataset_path, data_file)
    data_list, data_list_rev = [], []
    data_snap, data_snap_rev = [], []
    time_now = -1

    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')[:4]
            s, p, o, t = list(map(int, line))
            if time_now == -1:
                time_now = t
            elif time_now != t:
                data_list.append(data_snap)
                data_list_rev.append(data_snap_rev)
                data_snap, data_snap_rev = [], []
                time_now = t
            data_snap.append([s, p, o, t])
            data_snap_rev.append([o, p+rel_num, s, t])
        data_list.append(data_snap)
        data_list_rev.append(data_snap_rev)

    data_list_all = []
    for i in range(len(data_list)):
        data_list_all.append(data_list[i])
        data_list_all.append(data_list_rev[i])
    quad_list = [quad for sublist in data_list_all for quad in sublist]
    # quad_tensor = torch.tensor(quad_list).cuda()

    return quad_list


def get_quad_list_logcl(dataset_path: str, data_file: str, rel_num: int):
    data_path = os.path.join(dataset_path, data_file)
    data_list, data_list_rev = [], []
    data_snap, data_snap_rev = [], []
    time_now = -1

    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')[:4]
            s, p, o, t = list(map(int, line))
            if time_now == -1:
                time_now = t
            elif time_now != t:
                data_list.append(data_snap)
                data_list_rev.append(data_snap_rev)
                data_snap, data_snap_rev = [], []
                time_now = t
            data_snap.append([s, p, o, t])
            data_snap_rev.append([o, p+rel_num, s, t])
        data_list.append(data_snap)
        data_list_rev.append(data_snap_rev)

    data_list_all = []
    for i in range(len(data_list)):
        data_list_all.append(data_list[i])
    for i in range(len(data_list)):
        data_list_all.append(data_list_rev[i])
    quad_list = [quad for sublist in data_list_all for quad in sublist]
    # quad_tensor = torch.tensor(quad_list).cuda()

    return quad_list


def get_quad_num_list(dataset_path: str) -> List[int]:
    train_path = os.path.join(dataset_path, 'train.txt')
    valid_path = os.path.join(dataset_path, 'valid.txt')
    test_path = os.path.join(dataset_path, 'test.txt')

    train_quad_num = get_quad_num(train_path)
    valid_quad_num = get_quad_num(valid_path)
    test_quad_num = get_quad_num(test_path)
    quad_num_list = [train_quad_num, valid_quad_num, test_quad_num]

    return quad_num_list


def get_quad_num(data_path: str) -> int:
    with open(data_path, 'r') as file:
        lines = file.readlines()

    return len(lines)


def get_time_snap_list(dataset_path: str, data_file: str, rel_num: int) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
    data_path = os.path.join(dataset_path, data_file)
    data_list, data_list_rev = [], []
    data_snap, data_snap_rev = [], []
    time_now = -1

    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')[:4]
            s, p, o, t = list(map(int, line))
            if time_now == -1:
                time_now = t
            elif time_now != t:
                data_list.append(data_snap)
                data_list_rev.append(data_snap_rev)
                data_snap, data_snap_rev = [], []
                time_now = t
            data_snap.append([s, p, o, t])
            data_snap_rev.append([o, p+rel_num, s, t])
        data_list.append(data_snap)
        data_list_rev.append(data_snap_rev)

    return data_list, data_list_rev


def get_extend_time_snap_list(dataset_path: str, data_file: str) -> List[List[List[int]]]:
    data_path = os.path.join(dataset_path, data_file)
    data_list = []
    data_snap = []
    time_now = -1

    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')[:5]
            s, p, o, t = list(map(int, line[:4]))
            out_score = float(line[4])
            if time_now == -1:
                time_now = t
            elif time_now != t:
                data_list.append(data_snap)
                data_snap = []
                time_now = t
            data_snap.append([s, p, o, t, out_score])
        data_list.append(data_snap)

    return data_list


def get_quad_arr(data_snap_list: List[List[List[int]]]) -> np.ndarray:
    data_quad_list = [quad for snap in data_snap_list for quad in snap]
    data_quad_arr = np.array(data_quad_list)

    return data_quad_arr


def get_quad_tensor(data_snap_list: List[List[List[int]]]) -> torch.LongTensor:
    data_quad_list = [quad for snap in data_snap_list for quad in snap]
    data_quad_tensor = torch.tensor(data_quad_list)

    return data_quad_tensor


def get_one_dim_out_score_from_extend(dataset_path: str) -> List[float]:
    train_snap_list = get_extend_time_snap_list(dataset_path, data_file='train_extend.txt')
    valid_snap_list = get_extend_time_snap_list(dataset_path, data_file='valid_extend.txt')
    test_snap_list = get_extend_time_snap_list(dataset_path, data_file='test_extend.txt')

    all_snap_list = train_snap_list + valid_snap_list + test_snap_list
    all_quad_arr = get_quad_arr(all_snap_list)
    out_score = list(all_quad_arr[:, 4])

    return out_score


def get_two_dim_out_score_from_extend(dataset_path: str) -> None:
    train_snap_list = get_extend_time_snap_list(dataset_path, data_file='train_extend.txt')
    valid_snap_list = get_extend_time_snap_list(dataset_path, data_file='valid_extend.txt')
    test_snap_list = get_extend_time_snap_list(dataset_path, data_file='test_extend.txt')

    train_out_score = [[quad[4] for quad in snap] for snap in train_snap_list]
    valid_out_score = [[quad[4] for quad in snap] for snap in valid_snap_list]
    test_out_score = [[quad[4] for quad in snap] for snap in test_snap_list]
    out_score_dict = {'train_out_score': train_out_score, 'valid_out_score': valid_out_score, 'test_out_score': test_out_score}

    json_data_path = os.path.join(dataset_path, 'out_score.json')
    with open(json_data_path, 'w') as file:
        json.dump(out_score_dict, file)


def stat_ranks_not_outstanding(rank_list):
    hits = [1, 3, 10]
    rank_tensor = torch.tensor(rank_list).cuda()
    mr = torch.mean(rank_tensor.float())
    mrr = torch.mean(1.0 / rank_tensor.float())

    hit_scores = []
    for hit in hits:
        avg_count = torch.mean((rank_tensor <= hit).float())
        hit_scores.append(round(avg_count.item() * 100, 2))
    return (round(mr.item() * 100, 2), round(mrr.item() * 100, 2), hit_scores)


def stat_ranks_outstanding(rank_list, out_score, args):
    total_rank = torch.tensor(rank_list).cuda()
    out_bias = args['out_bias']

    test_out_score_tensor_original = torch.tensor(out_score).cuda()
    test_out_score_tensor = test_out_score_tensor_original + out_bias

    min_old = out_bias
    max_old = out_bias + 1.0
    min_new = out_bias
    max_new = out_bias + 1.0 * args['scale_factor']
    test_out_score_tensor = min_new + (max_new - min_new) * (test_out_score_tensor - min_old) / (max_old - min_old)

    test_out_score_sum = test_out_score_tensor.sum()
    weighted_score = test_out_score_tensor / test_out_score_sum

    hits = [1, 3, 10]

    mr = (weighted_score * total_rank.float()).sum()
    mrr = (weighted_score * (1.0 / total_rank.float())).sum()

    hit_scores = []
    for hit in hits:
        hit_value = (total_rank <= hit).float()
        weighted_count = (weighted_score * hit_value).sum()
        hit_scores.append(round(weighted_count.item() * 100, 2))

    # return (mr.item(), mrr.item(), hit_scores, mrr_snapshot_list)
    return (round(mr.item() * 100, 2), round(mrr.item() * 100, 2), hit_scores)


def get_sorted_out_score_from_extend(dataset_path: str, data_file: str) -> List[List[int]]:
    data_path = os.path.join(dataset_path, data_file)
    out_score_list = []
    data_snap_list = []
    data_snap = []
    data_snap_reverse = []
    time_now = -1

    _, rel_num = get_stat(dataset_path)

    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')[:5]
            s, p, o, t = list(map(int, line[:4]))
            out_score = float(line[4])
            if time_now == -1:
                time_now = t
            elif time_now != t:
                data_snap_sorted = sorted(data_snap, key=lambda x: x[1])
                data_snap_reverse_sorted = sorted(data_snap_reverse, key=lambda x: x[1])
                out_score_snap = [tuple[4] for tuple in data_snap_sorted]
                data_snap_list.append(data_snap_sorted)
                data_snap_list.append(data_snap_reverse_sorted)
                out_score_list.append(out_score_snap)
                out_score_list.append(out_score_snap)
                data_snap = []
                data_snap_reverse = []
                time_now = t
            data_snap.append([s, p, o, t, out_score])
            data_snap_reverse.append([o, p+rel_num, s, t, out_score])

        data_snap_sorted = sorted(data_snap, key=lambda x: x[1])
        data_snap_reverse_sorted = sorted(data_snap_reverse, key=lambda x: x[1])
        out_score_snap = [tuple[4] for tuple in data_snap_sorted]
        data_snap_list.append(data_snap_sorted)
        data_snap_list.append(data_snap_reverse_sorted)
        out_score_list.append(out_score_snap)
        out_score_list.append(out_score_snap)

    quad_list = [quad for sublist in data_snap_list for quad in sublist]
    out_score_list = [score for sublist in out_score_list for score in sublist]

    return quad_list, out_score_list


def get_sorted_out_score_from_extend_test(dataset_path: str, setting_name: str, data_file: str) -> List[List[int]]:
    data_path = os.path.join(dataset_path, setting_name, data_file)
    out_score_list = []
    data_snap_list = []
    data_snap = []
    data_snap_reverse = []
    time_now = -1

    _, rel_num = get_stat(dataset_path)

    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')[:5]
            s, p, o, t = list(map(int, line[:4]))
            out_score = float(line[4])
            if time_now == -1:
                time_now = t
            elif time_now != t:
                data_snap_sorted = sorted(data_snap, key=lambda x: x[1])
                data_snap_reverse_sorted = sorted(data_snap_reverse, key=lambda x: x[1])
                out_score_snap = [tuple[4] for tuple in data_snap_sorted]
                data_snap_list.append(data_snap_sorted)
                data_snap_list.append(data_snap_reverse_sorted)
                out_score_list.append(out_score_snap)
                out_score_list.append(out_score_snap)
                data_snap = []
                data_snap_reverse = []
                time_now = t
            data_snap.append([s, p, o, t, out_score])
            data_snap_reverse.append([o, p+rel_num, s, t, out_score])

        data_snap_sorted = sorted(data_snap, key=lambda x: x[1])
        data_snap_reverse_sorted = sorted(data_snap_reverse, key=lambda x: x[1])
        out_score_snap = [tuple[4] for tuple in data_snap_sorted]
        data_snap_list.append(data_snap_sorted)
        data_snap_list.append(data_snap_reverse_sorted)
        out_score_list.append(out_score_snap)
        out_score_list.append(out_score_snap)

    quad_list = [quad for sublist in data_snap_list for quad in sublist]
    out_score_list = [score for sublist in out_score_list for score in sublist]

    return quad_list, out_score_list


def ent_context_retrieval(case_quad, data_snap_list: List[List[List[int]]], rules_dict: Dict[int, List[Dict[str, Any]]], decay: int, his_len: int, reverse=False):
    dataset = 'ICEWS18'
    dataset_path = f'../data/{dataset}'
    ent2id = get_json_file(dataset_path, 'entity2id.json')
    rel2id = get_json_file(dataset_path, 'relation2id.json')
    ts2id = get_json_file(dataset_path, 'ts2id.json')
    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}
    id2ts = {v: k for k, v in ts2id.items()}

    ent_num, rel_num = get_stat(dataset_path)

    s, p, o, t = case_quad
    his_snaps = data_snap_list[t - his_len:t]
    his_quad_tensor = get_quad_tensor(his_snaps)

    cond_s = his_quad_tensor[:, 0] == s

    if p not in rules_dict:
        print("no rules for relation r\n")
        return

    rule_list = rules_dict[p]
    cond_p = his_quad_tensor[:, 1] == p
    cond_rel = cond_p.clone()
    body_rel_list = [rule['body_rels'][0] for rule in rule_list]
    for body_rel in body_rel_list:
        cond_body_rel = his_quad_tensor[:, 1] == body_rel
        cond_rel = cond_rel | cond_body_rel
    peer_ent = set(his_quad_tensor[cond_s & cond_rel, 2].tolist())
    if not peer_ent:
        print(f'no peer_ent for entity {id2ent[o]}\n')
        return

    peer_ent.add(o)
    peer_dict = {k: 0 for k in peer_ent}
    
    for rule in rule_list:
        body_rel = rule['body_rels'][0]
        rule_conf = rule['conf']
        cond_body = his_quad_tensor[:, 1] == body_rel

        for peer in peer_dict:
            cond_o = his_quad_tensor[:, 2] == peer
            ts_body = his_quad_tensor[cond_s & cond_body & cond_o, 3].tolist()
            ts_head = his_quad_tensor[cond_s & cond_p & cond_o, 3].tolist()
            if not ts_body:
                continue
            ts_body = sorted(ts_body, reverse=True)
            ts_head = sorted(ts_head, reverse=True)
            body_idx, head_idx = 0, 0
            len_body, len_head = len(ts_body), len(ts_head)

            while 1:
                peer_dict[peer] += rule_conf * math.exp(-1 * decay * (t - ts_body[body_idx]))
                if not ts_head:
                    break
                if body_rel == p:
                    while head_idx < len_head and ts_body[body_idx] <= ts_head[head_idx]:
                        head_idx += 1
                else:
                    while head_idx < len_head and ts_body[body_idx] < ts_head[head_idx]:
                        head_idx += 1
                if head_idx == len_head:
                    break
                body_idx += 1
                if body_idx == len_body:
                    break
                while body_idx < len_body and ts_head[head_idx] <= ts_body[body_idx]:
                    body_idx += 1
                if body_idx == len_body:
                    break

    key_tensor = torch.tensor(list(peer_dict.keys()))
    values_tensor = torch.tensor(list(peer_dict.values()))
    target_idx = torch.where(key_tensor == o)[0]
    if torch.sum(values_tensor**2) != 0:
        values_tensor_norm = values_tensor**2 / torch.sum(values_tensor**2)
    else:
        values_tensor_norm = values_tensor

    target_value = values_tensor_norm[target_idx]
    larger_tensor = values_tensor_norm[values_tensor_norm > target_value]
    target_outstanding = torch.sum(larger_tensor * (larger_tensor - target_value))
    print(f'Strikingness Score of entity {id2ent[o]}: {target_outstanding.item()}')

    sorted_peer_keys = sorted(peer_dict, key=lambda x: peer_dict[x], reverse=True)
    values_tensor_norm_sort, _ = torch.sort(values_tensor_norm, descending=True)
    for id_peer, peer in enumerate(sorted_peer_keys):
        if id_peer >= 3 and peer != o:
            continue
        print(f'No.{id_peer + 1} peer entity: {id2ent[peer]}\tTemporal Score: {peer_dict[peer]}\tNorm Score: {values_tensor_norm_sort[id_peer]}')
        rule_apply = 0
        for id_rule, rule in enumerate(rule_list):
            if rule_apply >= 3:
                break
            body_rel = rule['body_rels'][0]
            rule_conf = rule['conf']
            cond_body = his_quad_tensor[:, 1] == body_rel
            cond_o = his_quad_tensor[:, 2] == peer
            ts_body = his_quad_tensor[cond_s & cond_body & cond_o, 3].tolist()
            ts_head = his_quad_tensor[cond_s & cond_p & cond_o, 3].tolist()
            if not ts_body:
                continue
            rule_apply += 1
            print(f'  No.{id_rule + 1} rule Cond = {rule_conf}:')
            ts_body = sorted(ts_body, reverse=True)
            ts_head = sorted(ts_head, reverse=True)
            body_idx, head_idx = 0, 0
            len_body, len_head = len(ts_body), len(ts_head)
            
            flag = True
            id_grounding = 0
            while 1:
                if id_grounding >= 3:
                    break
                if flag:
                    time_head = t
                    flag = False
                else:
                    time_head = ts_head[head_idx]
                if reverse:
                    print(f'  {id2ent[s]}\t{id2rel[body_rel-rel_num]}\t{id2ent[peer]}\t{id2ts[ts_body[body_idx]]} --------> {id2ent[s]}\t{id2rel[p-rel_num]}\t{id2ent[peer]}\t{id2ts[time_head]}')
                else:
                    print(f'  {id2ent[s]}\t{id2rel[body_rel]}\t{id2ent[peer]}\t{id2ts[ts_body[body_idx]]} --------> {id2ent[s]}\t{id2rel[p]}\t{id2ent[peer]}\t{id2ts[time_head]}')
                if not ts_head:
                    break
                if body_rel == p:
                    while head_idx < len_head and ts_body[body_idx] <= ts_head[head_idx]:
                        head_idx += 1
                else:
                    while head_idx < len_head and ts_body[body_idx] < ts_head[head_idx]:
                        head_idx += 1
                if head_idx == len_head:
                    break
                body_idx += 1
                if body_idx == len_body:
                    break
                while body_idx < len_body and ts_head[head_idx] <= ts_body[body_idx]:
                    body_idx += 1
                if body_idx == len_body:
                    break
                id_grounding += 1
        
        print('')


def rel_context_retrieval(case_quad, data_snap_list: List[List[List[int]]], decay: int, his_len: int):
    dataset = 'ICEWS18'
    dataset_path = f'../data/{dataset}'
    ent2id = get_json_file(dataset_path, 'entity2id.json')
    rel2id = get_json_file(dataset_path, 'relation2id.json')
    ts2id = get_json_file(dataset_path, 'ts2id.json')
    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}
    id2ts = {v: k for k, v in ts2id.items()}

    ent_num, rel_num = get_stat(dataset_path)

    s, p, o, t = case_quad
    his_snaps = data_snap_list[t - his_len:t]
    his_quad_tensor = get_quad_tensor(his_snaps)

    cond_s = his_quad_tensor[:, 0] == s
    cond_o = his_quad_tensor[:, 2] == o
    peer_rel = set(his_quad_tensor[cond_s & cond_o, 1].tolist())
    if not peer_rel:
        print(f'no peer_relation for relation {id2rel[p]}\n')
        return

    peer_rel.add(p)
    peer_dict = {k: 0 for k in peer_rel}
    for peer in peer_dict:
        cond_peer = his_quad_tensor[:, 1] == peer
        ts_peer = his_quad_tensor[cond_s & cond_peer & cond_o, 3]
        score_peer = torch.sum(torch.exp(-1 * decay * (t - ts_peer)))
        peer_dict[peer] = float(score_peer)

    key_tensor = torch.tensor(list(peer_dict.keys()))
    values_tensor = torch.tensor(list(peer_dict.values()))
    target_idx = torch.where(key_tensor == p)[0]
    if torch.sum(values_tensor**2) != 0:
        values_tensor_norm = values_tensor**2 / torch.sum(values_tensor**2)
    else:
        values_tensor_norm = values_tensor

    target_value = values_tensor_norm[target_idx]
    larger_tensor = values_tensor_norm[values_tensor_norm > target_value]
    target_outstanding = torch.sum(larger_tensor * (larger_tensor - target_value))
    print(f'Strikingness Score of relation {id2rel[p]}: {target_outstanding.item()}')

    sorted_peer_keys = sorted(peer_dict, key=lambda x: peer_dict[x], reverse=True)
    values_tensor_norm_sort, _ = torch.sort(values_tensor_norm, descending=True)
    for id_peer, peer in enumerate(sorted_peer_keys):
        if id_peer >= 3 and peer != p:
            continue
        print(f'No.{id_peer + 1} peer relation: {id2rel[peer]}\tHIstorical Score: {peer_dict[peer]}\tNorm Score: {values_tensor_norm_sort[id_peer]}')
        cond_peer = his_quad_tensor[:, 1] == peer
        ts_peer = his_quad_tensor[cond_s & cond_peer & cond_o, 3].tolist()
        # score_peer = torch.sum(torch.exp(-1 * decay * (t - ts_peer)))
        # peer_dict[peer] = float(score_peer)
        ts_peer_reverse = sorted(ts_peer, reverse=True)
        for id_time, time_peer in enumerate(ts_peer_reverse):
            if id_time >= 2:
                break
            print(f'  {id2ent[s]}\t{id2rel[peer]}\t{id2ent[o]}\t{id2ts[time_peer]}')
        
        print('')


def calc_ent_score(case_quad, data_snap_list: List[List[List[int]]], rules_dict: Dict[int, List[Dict[str, Any]]], decay: int, his_len: int, reverse=False):
    dataset = 'ICEWS18'
    dataset_path = f'../data/{dataset}'
    ent2id = get_json_file(dataset_path, 'entity2id.json')
    rel2id = get_json_file(dataset_path, 'relation2id.json')
    ts2id = get_json_file(dataset_path, 'ts2id.json')
    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}
    id2ts = {v: k for k, v in ts2id.items()}

    ent_num, rel_num = get_stat(dataset_path)

    s, p, o, t = case_quad
    his_snaps = data_snap_list[t - his_len:t]
    his_quad_tensor = get_quad_tensor(his_snaps)

    cond_s = his_quad_tensor[:, 0] == s

    if p not in rules_dict:
        # print("no rules for relation r\n")
        return 1.0

    rule_list = rules_dict[p]
    cond_p = his_quad_tensor[:, 1] == p
    cond_rel = cond_p.clone()
    body_rel_list = [rule['body_rels'][0] for rule in rule_list]
    for body_rel in body_rel_list:
        cond_body_rel = his_quad_tensor[:, 1] == body_rel
        cond_rel = cond_rel | cond_body_rel
    peer_ent = set(his_quad_tensor[cond_s & cond_rel, 2].tolist())
    if not peer_ent:
        # print(f'no peer_ent for entity {id2ent[o]}\n')
        return 1.0

    peer_ent.add(o)
    peer_dict = {k: 0 for k in peer_ent}
    
    for rule in rule_list:
        body_rel = rule['body_rels'][0]
        rule_conf = rule['conf']
        cond_body = his_quad_tensor[:, 1] == body_rel

        for peer in peer_dict:
            cond_o = his_quad_tensor[:, 2] == peer
            ts_body = his_quad_tensor[cond_s & cond_body & cond_o, 3].tolist()
            ts_head = his_quad_tensor[cond_s & cond_p & cond_o, 3].tolist()
            if not ts_body:
                continue
            ts_body = sorted(ts_body, reverse=True)
            ts_head = sorted(ts_head, reverse=True)
            body_idx, head_idx = 0, 0
            len_body, len_head = len(ts_body), len(ts_head)

            while 1:
                peer_dict[peer] += rule_conf * math.exp(-1 * decay * (t - ts_body[body_idx]))
                if not ts_head:
                    break
                if body_rel == p:
                    while head_idx < len_head and ts_body[body_idx] <= ts_head[head_idx]:
                        head_idx += 1
                else:
                    while head_idx < len_head and ts_body[body_idx] < ts_head[head_idx]:
                        head_idx += 1
                if head_idx == len_head:
                    break
                body_idx += 1
                if body_idx == len_body:
                    break
                while body_idx < len_body and ts_head[head_idx] <= ts_body[body_idx]:
                    body_idx += 1
                if body_idx == len_body:
                    break

    key_tensor = torch.tensor(list(peer_dict.keys()))
    values_tensor = torch.tensor(list(peer_dict.values()))
    target_idx = torch.where(key_tensor == o)[0]
    if torch.sum(values_tensor**2) != 0:
        values_tensor_norm = values_tensor**2 / torch.sum(values_tensor**2)
    else:
        values_tensor_norm = values_tensor

    target_value = values_tensor_norm[target_idx]
    larger_tensor = values_tensor_norm[values_tensor_norm > target_value]
    target_outstanding = torch.sum(larger_tensor * (larger_tensor - target_value))
    target_outstanding = target_outstanding.item()
    
    return target_outstanding


def calc_rel_score(case_quad, data_snap_list: List[List[List[int]]], decay: int, his_len: int):
    dataset = 'ICEWS18'
    dataset_path = f'../data/{dataset}'
    ent2id = get_json_file(dataset_path, 'entity2id.json')
    rel2id = get_json_file(dataset_path, 'relation2id.json')
    ts2id = get_json_file(dataset_path, 'ts2id.json')
    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}
    id2ts = {v: k for k, v in ts2id.items()}

    ent_num, rel_num = get_stat(dataset_path)

    s, p, o, t = case_quad
    his_snaps = data_snap_list[t - his_len:t]
    his_quad_tensor = get_quad_tensor(his_snaps)

    cond_s = his_quad_tensor[:, 0] == s
    cond_o = his_quad_tensor[:, 2] == o
    peer_rel = set(his_quad_tensor[cond_s & cond_o, 1].tolist())
    if not peer_rel:
        # print(f'no peer_relation for relation {id2rel[p]}\n')
        return 1.0

    peer_rel.add(p)
    peer_dict = {k: 0 for k in peer_rel}
    for peer in peer_dict:
        cond_peer = his_quad_tensor[:, 1] == peer
        ts_peer = his_quad_tensor[cond_s & cond_peer & cond_o, 3]
        score_peer = torch.sum(torch.exp(-1 * decay * (t - ts_peer)))
        peer_dict[peer] = float(score_peer)

    key_tensor = torch.tensor(list(peer_dict.keys()))
    values_tensor = torch.tensor(list(peer_dict.values()))
    target_idx = torch.where(key_tensor == p)[0]
    if torch.sum(values_tensor**2) != 0:
        values_tensor_norm = values_tensor**2 / torch.sum(values_tensor**2)
    else:
        values_tensor_norm = values_tensor

    target_value = values_tensor_norm[target_idx]
    larger_tensor = values_tensor_norm[values_tensor_norm > target_value]
    target_outstanding = torch.sum(larger_tensor * (larger_tensor - target_value))
    target_outstanding = target_outstanding.item()

    return target_outstanding
    

def get_ent_context_lower(case_quad, data_snap_list: List[List[List[int]]], rules_dict: Dict[int, List[Dict[str, Any]]], decay: int, his_len: int, reverse=False):
    dataset = 'ICEWS18'
    dataset_path = f'../data/{dataset}'
    ent2id = get_json_file(dataset_path, 'entity2id.json')
    rel2id = get_json_file(dataset_path, 'relation2id.json')
    ts2id = get_json_file(dataset_path, 'ts2id.json')
    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}
    id2ts = {v: k for k, v in ts2id.items()}

    ent_num, rel_num = get_stat(dataset_path)

    s, p, o, t = case_quad
    his_snaps = data_snap_list[t - his_len:t]
    his_quad_tensor = get_quad_tensor(his_snaps)

    cond_s = his_quad_tensor[:, 0] == s

    if p not in rules_dict:
        print("no rules for relation r\n")
        return

    rule_list = rules_dict[p]
    cond_p = his_quad_tensor[:, 1] == p
    cond_rel = cond_p.clone()
    body_rel_list = [rule['body_rels'][0] for rule in rule_list]
    for body_rel in body_rel_list:
        cond_body_rel = his_quad_tensor[:, 1] == body_rel
        cond_rel = cond_rel | cond_body_rel
    peer_ent = set(his_quad_tensor[cond_s & cond_rel, 2].tolist())
    if not peer_ent:
        print(f'no peer_ent for entity {id2ent[o]}\n')
        return

    peer_ent.add(o)
    peer_dict = {k: 0 for k in peer_ent}
    
    for rule in rule_list:
        body_rel = rule['body_rels'][0]
        rule_conf = rule['conf']
        cond_body = his_quad_tensor[:, 1] == body_rel

        for peer in peer_dict:
            cond_o = his_quad_tensor[:, 2] == peer
            ts_body = his_quad_tensor[cond_s & cond_body & cond_o, 3].tolist()
            ts_head = his_quad_tensor[cond_s & cond_p & cond_o, 3].tolist()
            if not ts_body:
                continue
            ts_body = sorted(ts_body, reverse=True)
            ts_head = sorted(ts_head, reverse=True)
            body_idx, head_idx = 0, 0
            len_body, len_head = len(ts_body), len(ts_head)

            while 1:
                peer_dict[peer] += rule_conf * math.exp(-1 * decay * (t - ts_body[body_idx]))
                if not ts_head:
                    break
                if body_rel == p:
                    while head_idx < len_head and ts_body[body_idx] <= ts_head[head_idx]:
                        head_idx += 1
                else:
                    while head_idx < len_head and ts_body[body_idx] < ts_head[head_idx]:
                        head_idx += 1
                if head_idx == len_head:
                    break
                body_idx += 1
                if body_idx == len_body:
                    break
                while body_idx < len_body and ts_head[head_idx] <= ts_body[body_idx]:
                    body_idx += 1
                if body_idx == len_body:
                    break

    key_tensor = torch.tensor(list(peer_dict.keys()))
    values_tensor = torch.tensor(list(peer_dict.values()))
    target_idx = torch.where(key_tensor == o)[0]
    if torch.sum(values_tensor**2) != 0:
        values_tensor_norm = values_tensor**2 / torch.sum(values_tensor**2)
    else:
        values_tensor_norm = values_tensor

    target_value = values_tensor_norm[target_idx]
    larger_tensor = values_tensor_norm[values_tensor_norm > target_value]
    target_outstanding = torch.sum(larger_tensor * (larger_tensor - target_value)).item()

    sorted_peer_keys = sorted(peer_dict, key=lambda x: peer_dict[x], reverse=True)
    values_tensor_norm_sort, _ = torch.sort(values_tensor_norm, descending=False)
    values_tensor_norm_sort_lower = values_tensor_norm_sort[values_tensor_norm_sort < target_value]
    if values_tensor_norm_sort_lower.shape[0] < 1:
        return {}
    random_index = random.randint(0, values_tensor_norm_sort_lower.shape[0] - 1)

    # sorted_peer_keys = list(peer_dict.keys())
    # values_tensor_norm_sort_lower = values_tensor_norm[values_tensor_norm < target_value]
    # if values_tensor_norm_sort_lower.shape[0] < 1:
    #     return {}
    # random_index = random.randint(0, values_tensor_norm_sort_lower.shape[0] - 1)
    id_peer = random_index
    peer = sorted_peer_keys[id_peer]
    peer_context = []
    while 1:
        for id_rule, rule in enumerate(rule_list):
            body_rel = rule['body_rels'][0]
            rule_conf = rule['conf']
            cond_body = his_quad_tensor[:, 1] == body_rel
            cond_o = his_quad_tensor[:, 2] == peer
            ts_body = his_quad_tensor[cond_s & cond_body & cond_o, 3].tolist()
            ts_head = his_quad_tensor[cond_s & cond_p & cond_o, 3].tolist()
            if not ts_body:
                continue
            ts_body = sorted(ts_body, reverse=True)
            ts_head = sorted(ts_head, reverse=True)
            body_idx, head_idx = 0, 0
            len_body, len_head = len(ts_body), len(ts_head)
            
            flag = True
            id_grounding = 0
            while 1:
                if flag:
                    rule_grounding = [id2ent[s], id2rel[body_rel], id2ent[peer], id2ts[ts_body[body_idx]]]
                    peer_context.append(rule_grounding)
                    flag = False
                else:
                    time_head = ts_head[head_idx]
                # if reverse:
                #     print(f'  {id2ent[s]}\t{id2rel[body_rel-rel_num]}\t{id2ent[peer]}\t{id2ts[ts_body[body_idx]]} --------> {id2ent[s]}\t{id2rel[p-rel_num]}\t{id2ent[peer]}\t{id2ts[time_head]}')
                # else:
                    rule_body = [id2ent[s], id2rel[body_rel], id2ent[peer], id2ts[ts_body[body_idx]]]
                    rule_head = [id2ent[s], id2rel[p], id2ent[peer], id2ts[time_head]]
                    rule_grounding = [rule_body, rule_head]
                    peer_context.append(rule_grounding)
                    # print(f'  {id2ent[s]}\t{id2rel[body_rel]}\t{id2ent[peer]}\t{id2ts[ts_body[body_idx]]} --------> {id2ent[s]}\t{id2rel[p]}\t{id2ent[peer]}\t{id2ts[time_head]}')
                # peer_dict[peer] += rule_conf * math.exp(-1 * decay * (t - ts_body[body_idx]))
                if not ts_head:
                    break
                if body_rel == p:
                    while head_idx < len_head and ts_body[body_idx] <= ts_head[head_idx]:
                        head_idx += 1
                else:
                    while head_idx < len_head and ts_body[body_idx] < ts_head[head_idx]:
                        head_idx += 1
                if head_idx == len_head:
                    break
                body_idx += 1
                if body_idx == len_body:
                    break
                while body_idx < len_body and ts_head[head_idx] <= ts_body[body_idx]:
                    body_idx += 1
                if body_idx == len_body:
                    break
        if len(peer_context) == 0:
            random_index = random.randint(0, values_tensor_norm_sort_lower.shape[0] - 1)
            id_peer = random_index
            peer = sorted_peer_keys[id_peer]
        else:
            break

    # peer_value = values_tensor_norm[id_peer]  
    # larger_tensor = values_tensor_norm[values_tensor_norm > peer_value]
    # peer_outstanding = torch.sum(larger_tensor * (larger_tensor - peer_value)).item()
    
    target_event = [id2ent[s], id2rel[p], id2ent[o], id2ts[t]]
    peer_event = [id2ent[s], id2rel[p], id2ent[peer], id2ts[t]]

    # id_target = values_tensor_norm_sort_lower.shape[0] - 1
    peer = o
    target_context = []

    for id_rule, rule in enumerate(rule_list):
        body_rel = rule['body_rels'][0]
        rule_conf = rule['conf']
        cond_body = his_quad_tensor[:, 1] == body_rel
        cond_o = his_quad_tensor[:, 2] == peer
        ts_body = his_quad_tensor[cond_s & cond_body & cond_o, 3].tolist()
        ts_head = his_quad_tensor[cond_s & cond_p & cond_o, 3].tolist()
        if not ts_body:
            continue
        ts_body = sorted(ts_body, reverse=True)
        ts_head = sorted(ts_head, reverse=True)
        body_idx, head_idx = 0, 0
        len_body, len_head = len(ts_body), len(ts_head)
        
        flag = True
        id_grounding = 0
        while 1:
            if flag:
                rule_grounding = [id2ent[s], id2rel[body_rel], id2ent[peer], id2ts[ts_body[body_idx]]]
                target_context.append(rule_grounding)
                flag = False
            else:
                time_head = ts_head[head_idx]
            # if reverse:
            #     print(f'  {id2ent[s]}\t{id2rel[body_rel-rel_num]}\t{id2ent[peer]}\t{id2ts[ts_body[body_idx]]} --------> {id2ent[s]}\t{id2rel[p-rel_num]}\t{id2ent[peer]}\t{id2ts[time_head]}')
            # else:
                rule_body = [id2ent[s], id2rel[body_rel], id2ent[peer], id2ts[ts_body[body_idx]]]
                rule_head = [id2ent[s], id2rel[p], id2ent[peer], id2ts[time_head]]
                rule_grounding = [rule_body, rule_head]
                target_context.append(rule_grounding)
                # print(f'  {id2ent[s]}\t{id2rel[body_rel]}\t{id2ent[peer]}\t{id2ts[ts_body[body_idx]]} --------> {id2ent[s]}\t{id2rel[p]}\t{id2ent[peer]}\t{id2ts[time_head]}')
            # peer_dict[peer] += rule_conf * math.exp(-1 * decay * (t - ts_body[body_idx]))
            if not ts_head:
                break
            if body_rel == p:
                while head_idx < len_head and ts_body[body_idx] <= ts_head[head_idx]:
                    head_idx += 1
            else:
                while head_idx < len_head and ts_body[body_idx] < ts_head[head_idx]:
                    head_idx += 1
            if head_idx == len_head:
                break
            body_idx += 1
            if body_idx == len_body:
                break
            while body_idx < len_body and ts_head[head_idx] <= ts_body[body_idx]:
                body_idx += 1
            if body_idx == len_body:
                break

    case_dict = {'target_event': target_event, 'target_context': target_context, 
                 'peer_event': peer_event, 'peer_context': peer_context, 'target_outstanding': 0}

    return case_dict


def get_ent_context_higher(case_quad, data_snap_list: List[List[List[int]]], rules_dict: Dict[int, List[Dict[str, Any]]], decay: int, his_len: int, reverse=False):
    dataset = 'ICEWS18'
    dataset_path = f'../data/{dataset}'
    ent2id = get_json_file(dataset_path, 'entity2id.json')
    rel2id = get_json_file(dataset_path, 'relation2id.json')
    ts2id = get_json_file(dataset_path, 'ts2id.json')
    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}
    id2ts = {v: k for k, v in ts2id.items()}

    ent_num, rel_num = get_stat(dataset_path)

    s, p, o, t = case_quad
    his_snaps = data_snap_list[t - his_len:t]
    his_quad_tensor = get_quad_tensor(his_snaps)

    cond_s = his_quad_tensor[:, 0] == s

    if p not in rules_dict:
        # print("no rules for relation r\n")
        return {}

    rule_list = rules_dict[p]
    cond_p = his_quad_tensor[:, 1] == p
    cond_rel = cond_p.clone()
    body_rel_list = [rule['body_rels'][0] for rule in rule_list]
    for body_rel in body_rel_list:
        cond_body_rel = his_quad_tensor[:, 1] == body_rel
        cond_rel = cond_rel | cond_body_rel
    peer_ent = set(his_quad_tensor[cond_s & cond_rel, 2].tolist())
    if not peer_ent:
        # print(f'no peer_ent for entity {id2ent[o]}\n')
        return {}

    peer_ent.add(o)
    peer_dict = {k: 0 for k in peer_ent}
    
    for rule in rule_list:
        body_rel = rule['body_rels'][0]
        rule_conf = rule['conf']
        cond_body = his_quad_tensor[:, 1] == body_rel

        for peer in peer_dict:
            cond_o = his_quad_tensor[:, 2] == peer
            ts_body = his_quad_tensor[cond_s & cond_body & cond_o, 3].tolist()
            ts_head = his_quad_tensor[cond_s & cond_p & cond_o, 3].tolist()
            if not ts_body:
                continue
            ts_body = sorted(ts_body, reverse=True)
            ts_head = sorted(ts_head, reverse=True)
            body_idx, head_idx = 0, 0
            len_body, len_head = len(ts_body), len(ts_head)

            while 1:
                peer_dict[peer] += rule_conf * math.exp(-1 * decay * (t - ts_body[body_idx]))
                if not ts_head:
                    break
                if body_rel == p:
                    while head_idx < len_head and ts_body[body_idx] <= ts_head[head_idx]:
                        head_idx += 1
                else:
                    while head_idx < len_head and ts_body[body_idx] < ts_head[head_idx]:
                        head_idx += 1
                if head_idx == len_head:
                    break
                body_idx += 1
                if body_idx == len_body:
                    break
                while body_idx < len_body and ts_head[head_idx] <= ts_body[body_idx]:
                    body_idx += 1
                if body_idx == len_body:
                    break

    key_tensor = torch.tensor(list(peer_dict.keys()))
    values_tensor = torch.tensor(list(peer_dict.values()))
    target_idx = torch.where(key_tensor == o)[0]
    if torch.sum(values_tensor**2) != 0:
        values_tensor_norm = values_tensor**2 / torch.sum(values_tensor**2)
    else:
        values_tensor_norm = values_tensor

    target_value = values_tensor_norm[target_idx]
    larger_tensor = values_tensor_norm[values_tensor_norm > target_value]
    target_outstanding = torch.sum(larger_tensor * (larger_tensor - target_value)).item()

    sorted_peer_keys = sorted(peer_dict, key=lambda x: peer_dict[x], reverse=True)
    values_tensor_norm_sort, _ = torch.sort(values_tensor_norm, descending=True)
    values_tensor_norm_sort_lower = values_tensor_norm_sort[values_tensor_norm_sort > target_value]
    if values_tensor_norm_sort_lower.shape[0] < 1:
        return {}

    # target context
    # id_target = values_tensor_norm_sort_lower.shape[0] - 1
    peer = o
    target_context = []

    for id_rule, rule in enumerate(rule_list):
        body_rel = rule['body_rels'][0]
        rule_conf = rule['conf']
        cond_body = his_quad_tensor[:, 1] == body_rel
        cond_o = his_quad_tensor[:, 2] == peer
        ts_body = his_quad_tensor[cond_s & cond_body & cond_o, 3].tolist()
        ts_head = his_quad_tensor[cond_s & cond_p & cond_o, 3].tolist()
        if not ts_body:
            continue
        ts_body = sorted(ts_body, reverse=True)
        ts_head = sorted(ts_head, reverse=True)
        body_idx, head_idx = 0, 0
        len_body, len_head = len(ts_body), len(ts_head)
        
        flag = True
        id_grounding = 0
        while 1:
            if flag:
                rule_grounding = [id2ent[s], id2rel[body_rel], id2ent[peer], id2ts[ts_body[body_idx]]]
                target_context.append(rule_grounding)
                flag = False
            else:
                time_head = ts_head[head_idx]
            # if reverse:
            #     print(f'  {id2ent[s]}\t{id2rel[body_rel-rel_num]}\t{id2ent[peer]}\t{id2ts[ts_body[body_idx]]} --------> {id2ent[s]}\t{id2rel[p-rel_num]}\t{id2ent[peer]}\t{id2ts[time_head]}')
            # else:
                rule_body = [id2ent[s], id2rel[body_rel], id2ent[peer], id2ts[ts_body[body_idx]]]
                rule_head = [id2ent[s], id2rel[p], id2ent[peer], id2ts[time_head]]
                rule_grounding = [rule_body, rule_head]
                target_context.append(rule_grounding)
                # print(f'  {id2ent[s]}\t{id2rel[body_rel]}\t{id2ent[peer]}\t{id2ts[ts_body[body_idx]]} --------> {id2ent[s]}\t{id2rel[p]}\t{id2ent[peer]}\t{id2ts[time_head]}')
            # peer_dict[peer] += rule_conf * math.exp(-1 * decay * (t - ts_body[body_idx]))
            if not ts_head:
                break
            if body_rel == p:
                while head_idx < len_head and ts_body[body_idx] <= ts_head[head_idx]:
                    head_idx += 1
            else:
                while head_idx < len_head and ts_body[body_idx] < ts_head[head_idx]:
                    head_idx += 1
            if head_idx == len_head:
                break
            body_idx += 1
            if body_idx == len_body:
                break
            while body_idx < len_body and ts_head[head_idx] <= ts_body[body_idx]:
                body_idx += 1
            if body_idx == len_body:
                break
    
    if len(target_context) == 0:
        return {}

    # peer context
    random_index = random.randint(0, values_tensor_norm_sort_lower.shape[0] - 1)
    id_peer = random_index
    peer = sorted_peer_keys[id_peer]
    peer_context = []
    while 1:
        for id_rule, rule in enumerate(rule_list):
            body_rel = rule['body_rels'][0]
            rule_conf = rule['conf']
            cond_body = his_quad_tensor[:, 1] == body_rel
            cond_o = his_quad_tensor[:, 2] == peer
            ts_body = his_quad_tensor[cond_s & cond_body & cond_o, 3].tolist()
            ts_head = his_quad_tensor[cond_s & cond_p & cond_o, 3].tolist()
            if not ts_body:
                continue
            ts_body = sorted(ts_body, reverse=True)
            ts_head = sorted(ts_head, reverse=True)
            body_idx, head_idx = 0, 0
            len_body, len_head = len(ts_body), len(ts_head)
            
            flag = True
            id_grounding = 0
            while 1:
                if flag:
                    rule_grounding = [id2ent[s], id2rel[body_rel], id2ent[peer], id2ts[ts_body[body_idx]]]
                    peer_context.append(rule_grounding)
                    flag = False
                else:
                    time_head = ts_head[head_idx]
                # if reverse:
                #     print(f'  {id2ent[s]}\t{id2rel[body_rel-rel_num]}\t{id2ent[peer]}\t{id2ts[ts_body[body_idx]]} --------> {id2ent[s]}\t{id2rel[p-rel_num]}\t{id2ent[peer]}\t{id2ts[time_head]}')
                # else:
                    rule_body = [id2ent[s], id2rel[body_rel], id2ent[peer], id2ts[ts_body[body_idx]]]
                    rule_head = [id2ent[s], id2rel[p], id2ent[peer], id2ts[time_head]]
                    rule_grounding = [rule_body, rule_head]
                    peer_context.append(rule_grounding)
                    # print(f'  {id2ent[s]}\t{id2rel[body_rel]}\t{id2ent[peer]}\t{id2ts[ts_body[body_idx]]} --------> {id2ent[s]}\t{id2rel[p]}\t{id2ent[peer]}\t{id2ts[time_head]}')
                # peer_dict[peer] += rule_conf * math.exp(-1 * decay * (t - ts_body[body_idx]))
                if not ts_head:
                    break
                if body_rel == p:
                    while head_idx < len_head and ts_body[body_idx] <= ts_head[head_idx]:
                        head_idx += 1
                else:
                    while head_idx < len_head and ts_body[body_idx] < ts_head[head_idx]:
                        head_idx += 1
                if head_idx == len_head:
                    break
                body_idx += 1
                if body_idx == len_body:
                    break
                while body_idx < len_body and ts_head[head_idx] <= ts_body[body_idx]:
                    body_idx += 1
                if body_idx == len_body:
                    break
        if len(peer_context) == 0:
            random_index = random.randint(0, values_tensor_norm_sort_lower.shape[0] - 1)
            id_peer = random_index
            peer = sorted_peer_keys[id_peer]
        else:
            break

    peer_value = values_tensor_norm[id_peer]  
    larger_tensor = values_tensor_norm[values_tensor_norm > peer_value]
    peer_outstanding = torch.sum(larger_tensor * (larger_tensor - peer_value)).item()
    
    target_event = [id2ent[s], id2rel[p], id2ent[o], id2ts[t]]
    peer_event = [id2ent[s], id2rel[p], id2ent[peer], id2ts[t]]

    case_dict = {'target_event': target_event, 'target_context': target_context, 
                 'peer_event': peer_event, 'peer_context': peer_context, 'target_outstanding': 1}

    return case_dict
