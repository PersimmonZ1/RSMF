from typing import List, Dict, Any, Tuple
import argparse
import os
import json
import rule_application as ra
import numpy as np
import math
from tqdm import tqdm
from utils import get_stat, get_time_snap_list, get_quad_arr, get_two_dim_out_score_from_extend, get_quad_tensor
import torch
from datetime import datetime


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='ICEWS14')
    parser.add_argument('--seed', '-s', type=int, default=6)
    parser.add_argument('--rules_file', type=str, default='')
    parser.add_argument("--rule_lengths", "-l", type=int, default=1, nargs="+")
    parser.add_argument('--conf_threshold', type=float, default=0.3)
    parser.add_argument('--decay', type=float, default=0.1)
    parser.add_argument('--his_len', type=int, default=100)
    parser.add_argument('--obj_coeff', type=float, default=0.4)
    parser.add_argument('--sub_coeff', type=float, default=0.4)
    parser.add_argument('--rel_coeff', type=float, default=0.2)
    parser.add_argument('--out_threshold', type=float, default=0.5)
    parser.add_argument('--only_out_analysis', action='store_true', default=False)
    parser.add_argument('--save_out_score', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_false', default=True)

    args = parser.parse_args()

    return args


def calc_ent_out_score(data_snap_list: List[List[List[int]]], rules_dict: Dict[int, List[Dict[str, Any]]], decay: int, his_len: int) -> List[int]:
    out_score = []

    time_num = len(data_snap_list)
    for time_id in tqdm(range(time_num)):
        data_snap = data_snap_list[time_id]
        if time_id == 0:
            quad_num = len(data_snap)
            out_score.append([-1] * quad_num)
            continue
        out_score_snap = []
        if time_id <= his_len:
            his_snaps = data_snap_list[:time_id]
        else:
            his_snaps = data_snap_list[time_id - his_len:time_id]
        his_quad_arr = get_quad_arr(his_snaps)

        for quad in data_snap:
            s, p, o, t = quad
            cond_s = his_quad_arr[:, 0] == s
            cond_p = his_quad_arr[:, 1] == p
            peer_ent = set(his_quad_arr[cond_s & cond_p, 2])
            if not peer_ent or p not in rules_dict:  # 对等元素集合为空，或可利用规则集为空，认为是显著事件
                out_score_snap.append(1)
                continue

            peer_ent.add(o)
            peer_dict = {k: 0 for k in peer_ent}
            rule_list = rules_dict[p]
            # 先只考虑长度为1的规则
            for rule in rule_list:
                body_rel = rule['body_rels'][0]
                rule_conf = rule['conf']
                cond_body = his_quad_arr[:, 1] == body_rel

                for peer in peer_dict:
                    cond_o = his_quad_arr[:, 2] == peer
                    ts_body = list(his_quad_arr[cond_s & cond_body & cond_o, 3])
                    ts_head = list(his_quad_arr[cond_s & cond_p & cond_o, 3])
                    if not ts_body:  # 没有规则主体，该规则无效
                        continue
                    ts_body = sorted(ts_body, reverse=True)  # 倒序
                    ts_head = sorted(ts_head, reverse=True)
                    body_idx, head_idx = 0, 0
                    len_body, len_head = len(ts_body), len(ts_head)

                    # 计算规则主体和规则头部对的时态感知频次
                    while 1:
                        peer_dict[peer] += rule_conf * math.exp(-1 * decay * (t - ts_body[body_idx]))  # 第一步；对等元素显著评分计算
                        if not ts_head:  # 没有规则头部，无需进行后面的计算
                            break
                        if body_rel == p:  # 单独考虑重复事件
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

            # 第二步：计算归一化评分
            key_array = np.array(list(peer_dict.keys()))
            values_array = np.array(list(peer_dict.values()))
            target_idx = np.where(key_array == o)[0]
            if np.sum(values_array**2) != 0:
                values_array_norm = values_array**2 / np.sum(values_array**2)
            else:
                values_array_norm = values_array

            target_value = values_array_norm[target_idx]
            larger_array = values_array_norm[values_array_norm > target_value]
            target_outstanding = np.sum(larger_array * (larger_array - target_value))  # outlyingness函数
            out_score_snap.append(target_outstanding)
        out_score.append(out_score_snap)

    return out_score


def calc_rel_out_score(data_snap_list: List[List[List[int]]], decay: int, his_len: int) -> List[int]:
    out_score = []

    time_num = len(data_snap_list)
    for time_id in tqdm(range(time_num)):
        data_snap = data_snap_list[time_id]
        if time_id == 0:
            quad_num = len(data_snap)
            out_score.append([-1] * quad_num)
            continue
        out_score_snap = []
        if time_id <= his_len:
            his_snaps = data_snap_list[:time_id]
        else:
            his_snaps = data_snap_list[time_id - his_len:time_id]
        his_quad_arr = get_quad_arr(his_snaps)

        for quad in data_snap:
            s, p, o, t = quad
            cond_s = his_quad_arr[:, 0] == s
            cond_o = his_quad_arr[:, 2] == o
            peer_rel = set(his_quad_arr[cond_s & cond_o, 1])  # 对等元素集合
            if not peer_rel:  # 对等元素集合为空，认为是显著事件
                out_score_snap.append(1)
                continue

            peer_rel.add(p)
            peer_dict = {k: 0 for k in peer_rel}
            # 只考虑重复事实
            for peer in peer_dict:
                cond_peer = his_quad_arr[:, 1] == peer
                ts_peer = his_quad_arr[cond_s & cond_peer & cond_o, 3]
                score_peer = np.sum(np.exp(-1 * decay * (t - ts_peer)))
                peer_dict[peer] = float(score_peer)

            # 第二步：计算归一化评分
            key_array = np.array(list(peer_dict.keys()))
            values_array = np.array(list(peer_dict.values()))
            target_idx = np.where(key_array == p)[0]
            if np.sum(values_array**2) != 0:
                values_array_norm = values_array**2 / np.sum(values_array**2)
            else:
                values_array_norm = values_array

            target_value = values_array_norm[target_idx]
            larger_array = values_array_norm[values_array_norm > target_value]
            target_outstanding = np.sum(larger_array * (larger_array - target_value))  # outlyingness函数
            out_score_snap.append(target_outstanding)
        out_score.append(out_score_snap)

    return out_score


def calc_ent_out_score_cuda(data_snap_list: List[List[List[int]]], rules_dict: Dict[int, List[Dict[str, Any]]], decay: int, his_len: int) -> List[int]:
    out_score = []

    time_num = len(data_snap_list)
    for time_id in tqdm(range(time_num)):
        data_snap = data_snap_list[time_id]
        if time_id == 0:
            quad_num = len(data_snap)
            out_score.append([-1] * quad_num)
            continue
        out_score_snap = []
        if time_id <= his_len:
            his_snaps = data_snap_list[:time_id]
        else:
            his_snaps = data_snap_list[time_id - his_len:time_id]
        his_quad_tensor = get_quad_tensor(his_snaps)

        for quad in data_snap:
            s, p, o, t = quad
            cond_s = his_quad_tensor[:, 0] == s
            cond_p = his_quad_tensor[:, 1] == p
            peer_ent = set(his_quad_tensor[cond_s & cond_p, 2].tolist())  # 对等元素集合
            if not peer_ent or p not in rules_dict:  # 对等元素集合为空，或可利用规则集为空，认为是显著事件
                out_score_snap.append(1)
                continue

            peer_ent.add(o)
            peer_dict = {k: 0 for k in peer_ent}
            rule_list = rules_dict[p]
            # 先只考虑长度为1的规则
            for rule in rule_list:
                body_rel = rule['body_rels'][0]
                rule_conf = rule['conf']
                cond_body = his_quad_tensor[:, 1] == body_rel

                for peer in peer_dict:
                    cond_o = his_quad_tensor[:, 2] == peer
                    ts_body = his_quad_tensor[cond_s & cond_body & cond_o, 3].tolist()
                    ts_head = his_quad_tensor[cond_s & cond_p & cond_o, 3].tolist()
                    if not ts_body:  # 没有规则主体，该规则无效
                        continue
                    ts_body = sorted(ts_body, reverse=True)  # 倒序
                    ts_head = sorted(ts_head, reverse=True)
                    body_idx, head_idx = 0, 0
                    len_body, len_head = len(ts_body), len(ts_head)

                    # 计算规则主体和规则头部对的时态感知频次
                    while 1:
                        peer_dict[peer] += rule_conf * math.exp(-1 * decay * (t - ts_body[body_idx]))  # 第一步；对等元素显著评分计算
                        if not ts_head:  # 没有规则头部，无需进行后面的计算
                            break
                        if body_rel == p:  # 单独考虑重复事件
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

            # 第二步：计算归一化评分
            key_tensor = torch.tensor(list(peer_dict.keys()))
            values_tensor = torch.tensor(list(peer_dict.values()))
            target_idx = torch.where(key_tensor == o)[0]
            if torch.sum(values_tensor**2) != 0:
                values_tensor_norm = values_tensor**2 / torch.sum(values_tensor**2)
            else:
                values_tensor_norm = values_tensor

            target_value = values_tensor_norm[target_idx]
            larger_tensor = values_tensor_norm[values_tensor_norm > target_value]
            target_outstanding = torch.sum(larger_tensor * (larger_tensor - target_value))  # outlyingness函数
            out_score_snap.append(target_outstanding.item())
        out_score.append(out_score_snap)

    return out_score


def calc_rel_out_score_cuda(data_snap_list: List[List[List[int]]], decay: int, his_len: int) -> List[int]:
    out_score = []

    time_num = len(data_snap_list)
    for time_id in tqdm(range(time_num)):
        data_snap = data_snap_list[time_id]
        if time_id == 0:
            quad_num = len(data_snap)
            out_score.append([-1] * quad_num)
            continue
        out_score_snap = []
        if time_id <= his_len:
            his_snaps = data_snap_list[:time_id]
        else:
            his_snaps = data_snap_list[time_id - his_len:time_id]
        his_quad_tensor = get_quad_tensor(his_snaps)

        for quad in data_snap:
            s, p, o, t = quad
            cond_s = his_quad_tensor[:, 0] == s
            cond_o = his_quad_tensor[:, 2] == o
            peer_rel = set(his_quad_tensor[cond_s & cond_o, 1].tolist())  # 对等元素集合
            if not peer_rel:  # 对等元素集合为空，认为是显著事件
                out_score_snap.append(1)
                continue

            peer_rel.add(p)
            peer_dict = {k: 0 for k in peer_rel}
            # 只考虑重复事实
            for peer in peer_dict:
                cond_peer = his_quad_tensor[:, 1] == peer
                ts_peer = his_quad_tensor[cond_s & cond_peer & cond_o, 3]
                score_peer = torch.sum(torch.exp(-1 * decay * (t - ts_peer)))
                peer_dict[peer] = float(score_peer)

            # 第二步：计算归一化评分
            key_tensor = torch.tensor(list(peer_dict.keys()))
            values_tensor = torch.tensor(list(peer_dict.values()))
            target_idx = torch.where(key_tensor == p)[0]
            if torch.sum(values_tensor**2) != 0:
                values_tensor_norm = values_tensor**2 / torch.sum(values_tensor**2)
            else:
                values_tensor_norm = values_tensor

            target_value = values_tensor_norm[target_idx]
            larger_tensor = values_tensor_norm[values_tensor_norm > target_value]
            target_outstanding = torch.sum(larger_tensor * (larger_tensor - target_value))  # outlyingness函数
            out_score_snap.append(target_outstanding.item())
        out_score.append(out_score_snap)

    return out_score


def calc_rel_out_score_cuda_test(data_snap_list: List[List[List[int]]], decay: int, his_len: int, test_start_id: int) -> List[int]:
    out_score = []

    time_num = len(data_snap_list)
    for time_id in tqdm(range(test_start_id, time_num)):
        data_snap = data_snap_list[time_id]
        if time_id == 0:
            quad_num = len(data_snap)
            out_score.append([-1] * quad_num)
            continue
        out_score_snap = []
        if time_id <= his_len:
            his_snaps = data_snap_list[:time_id]
        else:
            his_snaps = data_snap_list[time_id - his_len:time_id]
        his_quad_tensor = get_quad_tensor(his_snaps)

        for quad in data_snap:
            s, p, o, t = quad
            cond_s = his_quad_tensor[:, 0] == s
            cond_o = his_quad_tensor[:, 2] == o
            peer_rel = set(his_quad_tensor[cond_s & cond_o, 1].tolist())  # 对等元素集合
            if not peer_rel:  # 对等元素集合为空，认为是显著事件
                out_score_snap.append(1)
                continue

            peer_rel.add(p)
            peer_dict = {k: 0 for k in peer_rel}
            # 只考虑重复事实
            for peer in peer_dict:
                cond_peer = his_quad_tensor[:, 1] == peer
                ts_peer = his_quad_tensor[cond_s & cond_peer & cond_o, 3]
                score_peer = torch.sum(torch.exp(-1 * decay * (t - ts_peer)))
                peer_dict[peer] = float(score_peer)

            # 第二步：计算归一化评分
            key_tensor = torch.tensor(list(peer_dict.keys()))
            values_tensor = torch.tensor(list(peer_dict.values()))
            target_idx = torch.where(key_tensor == p)[0]
            if torch.sum(values_tensor**2) != 0:
                values_tensor_norm = values_tensor**2 / torch.sum(values_tensor**2)
            else:
                values_tensor_norm = values_tensor

            target_value = values_tensor_norm[target_idx]
            larger_tensor = values_tensor_norm[values_tensor_norm > target_value]
            target_outstanding = torch.sum(larger_tensor * (larger_tensor - target_value))  # outlyingness函数
            out_score_snap.append(target_outstanding.item())
        out_score.append(out_score_snap)

    return out_score


def calc_ent_out_score_cuda_extend(data_snap_list: List[List[List[int]]], rules_dict: Dict[int, List[Dict[str, Any]]], decay: int, his_len: int) -> List[int]:
    out_score = []

    time_num = len(data_snap_list)
    for time_id in tqdm(range(time_num)):
        data_snap = data_snap_list[time_id]
        if time_id == 0:
            quad_num = len(data_snap)
            out_score.append([-1] * quad_num)
            continue
        out_score_snap = []
        if time_id <= his_len:
            his_snaps = data_snap_list[:time_id]
        else:
            his_snaps = data_snap_list[time_id - his_len:time_id]
        his_quad_tensor = get_quad_tensor(his_snaps)

        for quad in data_snap:
            s, p, o, t = quad
            cond_s = his_quad_tensor[:, 0] == s

            if p not in rules_dict:  # 可利用规则集为空，认为是显著事件
                out_score_snap.append(1)
                continue

            # 根据所有支撑规则统计对等元素集合
            rule_list = rules_dict[p]
            cond_p = his_quad_tensor[:, 1] == p
            cond_rel = cond_p.clone()
            body_rel_list = [rule['body_rels'][0] for rule in rule_list]
            for body_rel in body_rel_list:
                cond_body_rel = his_quad_tensor[:, 1] == body_rel
                cond_rel = cond_rel | cond_body_rel
            peer_ent = set(his_quad_tensor[cond_s & cond_rel, 2].tolist())  # 对等元素集合
            if not peer_ent:  # 对等元素集合为空，认为是显著事件
                out_score_snap.append(1)
                continue

            peer_ent.add(o)
            peer_dict = {k: 0 for k in peer_ent}
            # 先只考虑长度为1的规则
            for rule in rule_list:
                body_rel = rule['body_rels'][0]
                rule_conf = rule['conf']
                cond_body = his_quad_tensor[:, 1] == body_rel

                for peer in peer_dict:
                    cond_o = his_quad_tensor[:, 2] == peer
                    ts_body = his_quad_tensor[cond_s & cond_body & cond_o, 3].tolist()
                    ts_head = his_quad_tensor[cond_s & cond_p & cond_o, 3].tolist()
                    if not ts_body:  # 没有规则主体，该规则无效
                        continue
                    ts_body = sorted(ts_body, reverse=True)  # 倒序
                    ts_head = sorted(ts_head, reverse=True)
                    body_idx, head_idx = 0, 0
                    len_body, len_head = len(ts_body), len(ts_head)

                    # 计算规则主体和规则头部对的时态感知频次
                    while 1:
                        peer_dict[peer] += rule_conf * math.exp(-1 * decay * (t - ts_body[body_idx]))  # 第一步；对等元素显著评分计算
                        if not ts_head:  # 没有规则头部，无需进行后面的计算
                            break
                        if body_rel == p:  # 单独考虑重复事件
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

            # 第二步：计算归一化评分
            key_tensor = torch.tensor(list(peer_dict.keys()))
            values_tensor = torch.tensor(list(peer_dict.values()))
            target_idx = torch.where(key_tensor == o)[0]
            if torch.sum(values_tensor**2) != 0:
                values_tensor_norm = values_tensor**2 / torch.sum(values_tensor**2)
            else:
                values_tensor_norm = values_tensor

            target_value = values_tensor_norm[target_idx]
            larger_tensor = values_tensor_norm[values_tensor_norm > target_value]
            target_outstanding = torch.sum(larger_tensor * (larger_tensor - target_value))  # outlyingness函数
            out_score_snap.append(target_outstanding.item())
        out_score.append(out_score_snap)

    return out_score


def calc_ent_out_score_cuda_extend_test(data_snap_list: List[List[List[int]]], rules_dict: Dict[int, List[Dict[str, Any]]], decay: int, his_len: int, test_start_id: int) -> List[int]:
    out_score = []

    time_num = len(data_snap_list)
    for time_id in tqdm(range(test_start_id, time_num)):
        data_snap = data_snap_list[time_id]
        if time_id == 0:
            quad_num = len(data_snap)
            out_score.append([-1] * quad_num)
            continue
        out_score_snap = []
        if time_id <= his_len:
            his_snaps = data_snap_list[:time_id]
        else:
            his_snaps = data_snap_list[time_id - his_len:time_id]
        his_quad_tensor = get_quad_tensor(his_snaps)

        for quad in data_snap:
            s, p, o, t = quad
            cond_s = his_quad_tensor[:, 0] == s

            if p not in rules_dict:  # 可利用规则集为空，认为是显著事件
                out_score_snap.append(1)
                continue

            # 根据所有支撑规则统计对等元素集合
            rule_list = rules_dict[p]
            cond_p = his_quad_tensor[:, 1] == p
            cond_rel = cond_p.clone()
            body_rel_list = [rule['body_rels'][0] for rule in rule_list]
            for body_rel in body_rel_list:
                cond_body_rel = his_quad_tensor[:, 1] == body_rel
                cond_rel = cond_rel | cond_body_rel
            peer_ent = set(his_quad_tensor[cond_s & cond_rel, 2].tolist())  # 对等元素集合
            if not peer_ent:  # 对等元素集合为空，认为是显著事件
                out_score_snap.append(1)
                continue

            peer_ent.add(o)
            peer_dict = {k: 0 for k in peer_ent}
            # 先只考虑长度为1的规则
            for rule in rule_list:
                body_rel = rule['body_rels'][0]
                rule_conf = rule['conf']
                cond_body = his_quad_tensor[:, 1] == body_rel

                for peer in peer_dict:
                    cond_o = his_quad_tensor[:, 2] == peer
                    ts_body = his_quad_tensor[cond_s & cond_body & cond_o, 3].tolist()
                    ts_head = his_quad_tensor[cond_s & cond_p & cond_o, 3].tolist()
                    if not ts_body:  # 没有规则主体，该规则无效
                        continue
                    ts_body = sorted(ts_body, reverse=True)  # 倒序
                    ts_head = sorted(ts_head, reverse=True)
                    body_idx, head_idx = 0, 0
                    len_body, len_head = len(ts_body), len(ts_head)

                    # 计算规则主体和规则头部对的时态感知频次
                    while 1:
                        peer_dict[peer] += rule_conf * math.exp(-1 * decay * (t - ts_body[body_idx]))  # 第一步；对等元素显著评分计算
                        if not ts_head:  # 没有规则头部，无需进行后面的计算
                            break
                        if body_rel == p:  # 单独考虑重复事件
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

            # 第二步：计算归一化评分
            key_tensor = torch.tensor(list(peer_dict.keys()))
            values_tensor = torch.tensor(list(peer_dict.values()))
            target_idx = torch.where(key_tensor == o)[0]
            if torch.sum(values_tensor**2) != 0:
                values_tensor_norm = values_tensor**2 / torch.sum(values_tensor**2)
            else:
                values_tensor_norm = values_tensor

            target_value = values_tensor_norm[target_idx]
            larger_tensor = values_tensor_norm[values_tensor_norm > target_value]
            target_outstanding = torch.sum(larger_tensor * (larger_tensor - target_value))  # outlyingness函数
            out_score_snap.append(target_outstanding.item())
        out_score.append(out_score_snap)

    return out_score


def update_out_score_to_dataset(dataset_path: str, setting_name: str, out_score: List[List[float]]) -> None:
    out_score = [score for sublist in out_score for score in sublist]
    train_input_path = os.path.join(dataset_path, 'train.txt')
    train_output_path = os.path.join(dataset_path, setting_name, 'train_extend.txt')
    valid_input_path = os.path.join(dataset_path, 'valid.txt')
    valid_output_path = os.path.join(dataset_path, setting_name, 'valid_extend.txt')
    test_input_path = os.path.join(dataset_path, 'test.txt')
    test_output_path = os.path.join(dataset_path, setting_name, 'test_extend.txt')
    idx = 0

    with open(train_input_path, 'r') as input_file, open(train_output_path, 'w') as output_file:
        for input_line in input_file:
            output_line = input_line.strip() + f'\t{str(out_score[idx])[:5]}\n'
            output_file.write(output_line)
            idx += 1
    with open(valid_input_path, 'r') as input_file, open(valid_output_path, 'w') as output_file:
        for input_line in input_file:
            output_line = input_line.strip() + f'\t{str(out_score[idx])[:5]}\n'
            output_file.write(output_line)
            idx += 1
    with open(test_input_path, 'r') as input_file, open(test_output_path, 'w') as output_file:
        for input_line in input_file:
            output_line = input_line.strip() + f'\t{str(out_score[idx])[:5]}\n'
            output_file.write(output_line)
            idx += 1


def update_out_score_to_dataset_test(dataset_path: str, setting_name: str, out_score: List[List[float]]) -> None:
    out_score = [score for sublist in out_score for score in sublist]
    test_input_path = os.path.join(dataset_path, 'test.txt')
    test_output_path = os.path.join(dataset_path, setting_name, 'test_extend.txt')
    idx = 0

    with open(test_input_path, 'r') as input_file, open(test_output_path, 'w') as output_file:
        for input_line in input_file:
            output_line = input_line.strip() + f'\t{str(out_score[idx])[:5]}\n'
            output_file.write(output_line)
            idx += 1


def save_hyperparam(dataset_path: str, args: argparse.Namespace, out_score_dict: Dict[str, List[List[float]]], time_list=None) -> None:
    data_path = os.path.join(dataset_path, 'hyperparam.txt')
    if not os.path.exists(data_path):
        with open(data_path, 'w') as file:
            hyperparam_info = ('conf_threshold\tdeacy\this_len\tobj_coeff\tsub_coeff\trel_coeff\tout_threshold\t'
                               'train_bp\ttrain_ap\tvalid_bp\tvalid_ap\ttest_bp\ttest_ap\t\n')
            file.write(hyperparam_info)

    train_below_prop, train_above_prop = outstanding_analysis(out_score=out_score_dict['train_out_score'], out_threshold=args.out_threshold)
    valid_below_prop, valid_above_prop = outstanding_analysis(out_score=out_score_dict['valid_out_score'], out_threshold=args.out_threshold)
    test_below_prop, test_above_prop = outstanding_analysis(out_score=out_score_dict['test_out_score'], out_threshold=args.out_threshold)

    if not args.only_out_analysis:
        time_obj = str(time_list[0])[:7]
        time_sub = str(time_list[1])[:7]
        time_rel = str(time_list[2])[:7]
        time_sum = str(time_list[0] + time_list[1] + time_list[2])[:7]
    else:
        time_obj = time_sub = time_rel = time_sum = 0

    with open(data_path, 'a') as file:
        hyperparam_line = (f'{args.conf_threshold}\t\t\t{args.decay}\t\t\t{args.his_len}\t\t\t{args.obj_coeff}\t\t\t{args.sub_coeff}\t\t\t{args.rel_coeff}\t\t\t{args.out_threshold}\t\t\t'
                           f'{train_below_prop}\t\t{train_above_prop}\t\t{valid_below_prop}\t\t{valid_above_prop}\t\t{test_below_prop}\t\t{test_above_prop}\t\t\n'
                           f'{time_obj}+{time_sub}+{time_rel}={time_sum}\n')
        file.write(hyperparam_line)


def save_hyperparam_test(dataset_path: str, args: argparse.Namespace, out_score_dict: Dict[str, List[List[float]]], time_list=None) -> None:
    data_path = os.path.join(dataset_path, 'hyperparam_test.txt')
    if not os.path.exists(data_path):
        with open(data_path, 'w') as file:
            hyperparam_info = ('conf_threshold\tdeacy\this_len\tobj_coeff\tsub_coeff\trel_coeff\tout_threshold\t'
                               'test_bp\ttest_ap\t\n')
            file.write(hyperparam_info)

    # train_below_prop, train_above_prop = outstanding_analysis(out_score=out_score_dict['train_out_score'], out_threshold=args.out_threshold)
    # valid_below_prop, valid_above_prop = outstanding_analysis(out_score=out_score_dict['valid_out_score'], out_threshold=args.out_threshold)
    test_below_prop, test_above_prop = outstanding_analysis(out_score=out_score_dict['test_out_score'], out_threshold=args.out_threshold)

    if not args.only_out_analysis:
        time_obj = str(time_list[0])[:7]
        time_sub = str(time_list[1])[:7]
        time_rel = str(time_list[2])[:7]
        time_sum = str(time_list[0] + time_list[1] + time_list[2])[:7]
    else:
        time_obj = time_sub = time_rel = time_sum = 0

    with open(data_path, 'a') as file:
        hyperparam_line = (f'{args.conf_threshold}\t\t\t{args.decay}\t\t\t{args.his_len}\t\t\t{args.obj_coeff}\t\t\t{args.sub_coeff}\t\t\t{args.rel_coeff}\t\t\t{args.out_threshold}\t\t\t'
                           f'{test_below_prop}\t\t{test_above_prop}\t\t\n'
                           f'{time_obj}+{time_sub}+{time_rel}={time_sum}\n')
        file.write(hyperparam_line)


def outstanding_analysis(out_score: List[List[float]], out_threshold: float) -> Tuple[float, float]:
    out_score_one_dim = [score for sublist in out_score for score in sublist]
    out_score_tensor = torch.tensor(out_score_one_dim).cuda()

    below_threshold_mask = out_score_tensor < out_threshold
    above_threshold_mask = out_score_tensor >= out_threshold

    below_count = below_threshold_mask.sum()
    above_count = above_threshold_mask.sum()
    total_count = out_score_tensor.size(0)

    below_prop = round((below_count * 1.0 / total_count).item(), 3)
    above_prop = round((above_count * 1.0 / total_count).item(), 3)

    return below_prop, above_prop


if __name__ == '__main__':
    args = get_args()

    dataset_path = f'../data/{args.dataset}'

    ent_num, rel_num = get_stat(dataset_path)

    if args.only_out_analysis:  # 如果已经生成扩展后的数据集文件，则只要提取显著性评分即可
        setting_name = f'{args.conf_threshold}_{args.decay}_{args.his_len}_{args.obj_coeff}_{args.sub_coeff}_{args.rel_coeff}'
        out_score_file_path = os.path.join(dataset_path, setting_name, 'out_score_test.json')
        with open(out_score_file_path, 'r') as file:
            out_score_dict = json.load(file)
        test_out_score = out_score_dict['test_out_score']
        update_out_score_to_dataset_test(dataset_path, setting_name, test_out_score)
    else:
        # 得到所有样本及反向四元组
        train_snap_list, train_snap_list_reverse = get_time_snap_list(dataset_path, data_file='train.txt', rel_num=rel_num)
        valid_snap_list, valid_snap_list_reverse = get_time_snap_list(dataset_path, data_file='valid.txt', rel_num=rel_num)
        test_snap_list, test_snap_list_reverse = get_time_snap_list(dataset_path, data_file='test.txt', rel_num=rel_num)
        all_snap_list = train_snap_list + valid_snap_list + test_snap_list
        all_snap_list_reverse = train_snap_list_reverse + valid_snap_list_reverse + test_snap_list_reverse

        # 得到规则
        rules_path = f'../output_rule/{args.dataset}/seed{args.seed}/{args.rules_file}'
        rule_lengths = args.rule_lengths
        rule_lengths = [rule_lengths] if (rule_lengths is int) else rule_lengths
        rules_dict = json.load(open(rules_path))
        rules_dict = {int(k): v for k, v in rules_dict.items()}
        rules_dict = ra.filter_rules(  # 按照最小置信度和主体支持过滤规则，并删除没有规则后的键值对
            rules_dict, min_conf=args.conf_threshold, min_body_supp=2, rule_lengths=rule_lengths
        )

        test_start_id = len(train_snap_list) + len(valid_snap_list)
        # 计算每个样本的显著性评分，限制历史时间戳长度
        if args.cuda:
            start_time_obj = datetime.now()
            print('calculate objects\' outstanding score...')
            out_score_obj = calc_ent_out_score_cuda_extend_test(data_snap_list=all_snap_list, rules_dict=rules_dict, decay=args.decay, his_len=args.his_len, test_start_id=test_start_id)
            end_time_obj = datetime.now()
            time_obj = end_time_obj - start_time_obj

            start_time_sub = datetime.now()
            print('calculate subjects\' outstanding score...')
            out_score_sub = calc_ent_out_score_cuda_extend_test(data_snap_list=all_snap_list_reverse, rules_dict=rules_dict, decay=args.decay, his_len=args.his_len, test_start_id=test_start_id)
            end_time_sub = datetime.now()
            time_sub = end_time_sub - start_time_sub

            start_time_rel = datetime.now()
            print('calculate relations\' outstanding score...')
            out_score_rel = calc_rel_out_score_cuda_test(data_snap_list=all_snap_list, decay=args.decay, his_len=args.his_len, test_start_id=test_start_id)
            end_time_rel = datetime.now()
            time_rel = end_time_rel - start_time_rel
            time_list = [time_obj, time_sub, time_rel]
        else:
            print('calculate objects\' outstanding score...')
            out_score_obj = calc_ent_out_score(data_snap_list=all_snap_list, rules_dict=rules_dict, decay=args.decay, his_len=args.his_len)
            print('calculate subjects\' outstanding score...')
            out_score_sub = calc_ent_out_score(data_snap_list=all_snap_list_reverse, rules_dict=rules_dict, decay=args.decay, his_len=args.his_len)
            print('calculate relations\' outstanding score...')
            out_score_rel = calc_rel_out_score(data_snap_list=all_snap_list, decay=args.decay, his_len=args.his_len)

        print('weighting the outstanding score...')
        out_score = []
        time_num = len(out_score_obj)
        for time_id in tqdm(range(time_num)):
            out_score_snap = [args.obj_coeff * x + args.sub_coeff * y + args.rel_coeff * z
                              for x, y, z in zip(out_score_obj[time_id], out_score_sub[time_id], out_score_rel[time_id])]
            out_score.append(out_score_snap)
        out_score = [[round(score, 3) for score in sublist] for sublist in out_score]
        out_score_dict = {'test_out_score': out_score}

        if args.save_out_score:
            setting_name = f'{args.conf_threshold}_{args.decay}_{args.his_len}_{args.obj_coeff}_{args.sub_coeff}_{args.rel_coeff}'
            os.makedirs(os.path.join(dataset_path, setting_name), exist_ok=True)
            json_data_path = os.path.join(dataset_path, setting_name, 'out_score_test.json')
            with open(json_data_path, 'w') as file:
                json.dump(out_score_dict, file)
            update_out_score_to_dataset_test(dataset_path, setting_name, out_score)
