import json
import numpy as np


class Grapher(object):
    def __init__(self, dataset_dir):
        """
        Store information about the graph (train/valid/test set).
        Add corresponding inverse quadruples to the data.

        Parameters:
            dataset_dir (str): path to the graph dataset directory

        Returns:
            None
        """

        self.dataset_dir = dataset_dir
        self.entity2id = json.load(open(dataset_dir + "entity2id.json", encoding="utf-8"))
        self.relation2id_old = json.load(open(dataset_dir + "relation2id.json", encoding="utf-8"))
        self.relation2id = self.relation2id_old.copy()
        counter = len(self.relation2id_old)
        for relation in self.relation2id_old:
            self.relation2id["_" + relation] = counter  # Inverse relation
            counter += 1
        self.ts2id = json.load(open(dataset_dir + "ts2id.json", encoding="utf-8"))
        self.id2entity = dict([(v, k) for k, v in self.entity2id.items()])
        self.id2relation = dict([(v, k) for k, v in self.relation2id.items()])
        self.id2ts = dict([(v, k) for k, v in self.ts2id.items()])

        self.inv_relation_id = dict()  # 存储id为i的关系对应的反向关系的id
        num_relations = len(self.relation2id_old)
        for i in range(num_relations):
            self.inv_relation_id[i] = i + num_relations
        for i in range(num_relations, num_relations * 2):
            self.inv_relation_id[i] = i % num_relations

        self.train_idx = self.create_store("train_raw.txt")  # 二维id数组
        self.valid_idx = self.create_store("valid_raw.txt")
        self.test_idx = self.create_store("test_raw.txt")
        self.all_idx = np.vstack((self.train_idx, self.valid_idx, self.test_idx))

        print("Grapher initialized.")

    def create_store(self, file):
        """
        Store the quadruples from the file as indices.
        The quadruples in the file should be in the format "subject\trelation\tobject\ttimestamp\n".

        Parameters:
            file (str): file name

        Returns:
            store_idx (np.ndarray): indices of quadruples
        """

        with open(self.dataset_dir + file, "r", encoding="utf-8") as f:
            quads = f.readlines()
        store = self.split_quads(quads)  # 得到四元组列表
        store_idx = self.map_to_idx(store)  # 得到转换为id后的四元组
        store_idx = self.add_inverses(store_idx)  # 得到反向四元组，注意相同时间戳的反向四元组并不在一起，后续可能要修改

        return store_idx

    def split_quads(self, quads):
        """
        Split quadruples into a list of strings.

        Parameters:
            quads (list): list of quadruples
                          Each quadruple has the form "subject\trelation\tobject\ttimestamp\n".

        Returns:
            split_q (list): list of quadruples
                            Each quadruple has the form [subject, relation, object, timestamp].
        """

        split_q = []
        for quad in quads:
            split_q.append(quad[:-1].split("\t"))

        return split_q

    def map_to_idx(self, quads):
        """
        Map quadruples to their indices.

        Parameters:
            quads (list): list of quadruples
                          Each quadruple has the form [subject, relation, object, timestamp].

        Returns:
            quads (np.ndarray): indices of quadruples
        """

        subs = [self.entity2id[x[0]] for x in quads]
        rels = [self.relation2id[x[1]] for x in quads]
        objs = [self.entity2id[x[2]] for x in quads]
        tss = [self.ts2id[x[3]] for x in quads]
        quads = np.column_stack((subs, rels, objs, tss))

        return quads

    def add_inverses(self, quads_idx):
        """
        Add the inverses of the quadruples as indices.

        Parameters:
            quads_idx (np.ndarray): indices of quadruples

        Returns:
            quads_idx (np.ndarray): indices of quadruples along with the indices of their inverses
        """

        quad_list = []
        start_idx, end_idx = 0, 0
        time_now = quads_idx[0, 3]
        for idx, quad in enumerate(quads_idx):
            if quad[3] != time_now:
                end_idx = idx
                quad_list.append(quads_idx[start_idx:end_idx])
                start_idx = idx
                time_now = quad[3]
        quad_list.append(quads_idx[start_idx:])

        for idx, snap in enumerate(quad_list):
            inv_snap = snap[:, [2, 1, 0, 3]]
            rel = [self.inv_relation_id[x] for x in snap[:, 1]]
            inv_snap[np.arange(len(inv_snap)), 1] = rel
            quad_list[idx] = np.vstack((snap, inv_snap))

        quad_idx = np.vstack(quad_list)

        # 相同时间戳的正方向四元组不在一起
        # subs = quads_idx[:, 2]
        # rels = [self.inv_relation_id[x] for x in quads_idx[:, 1]]
        # objs = quads_idx[:, 0]
        # tss = quads_idx[:, 3]
        # inv_quads_idx = np.column_stack((subs, rels, objs, tss))
        # quads_idx = np.vstack((quads_idx, inv_quads_idx))

        return quad_idx
