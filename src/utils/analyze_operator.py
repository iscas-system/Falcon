import os,sys,platform
import csv

gputype_list = ['M60','K80','T4','V100']
current_dir = os.path.abspath('.')
all_m = ['bert-large','densenet-201','gru','inception-v2','inception-v4','mobilenet-v2','resnet-101','resnet-152-v2','roberta','tacotron2','transformer','vgg16']
all_o = ['add','batch_norm','concat','conv1d','conv2d','dense','multiply','relu','sigmoid','split','strided_slice','tanh','transpose']
baselines = ['morphling', 'vesta', 'heterbo', 'ernest']

import json
import os
import platform
import sys
import csv
# from time import sleep, time
import pandas as pd
import numpy as np
# from torch import manual_seed
# from bayesmark.np_util import argmin_2d, random

# from bayesmark.constants import ITER
# import bayesmark.random_search as rs
# from bayesmark.experiment import logger

from bayesmark.space import JointSpace
# from bayesmark.stats import robust_standardize
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# from utils import sampler
# from utils.turbo1 import TuRBO1
# from turbo.utils import from_unit_cube, latin_hypercube, to_unit_cube
from copy import deepcopy
from typing import Optional
# try:
#     from util import copula_standardize
# except:
#     from utils.util import copula_standardize
try:
    from utils import query, standardize, devision
except:
    import query, standardize, devision

all_m = ['bert-large', 'densenet-201', 'gru', 'inception-v2', 'inception-v4', 'mobilenet-v2', 'resnet-101',
         'resnet-152-v2', 'roberta', 'tacotron2', 'transformer', 'vgg16']
all_gputype = ['M60','K80','T4','V100']
current_dir = os.path.abspath('.')

X_list = []
y_list = []
islast = False
model = ""
operator = ""
max_margin_labels = None
optimal_points_in_operator_data = None
near_optimal_points_in_operator_data = None
iteration = 0

api_config = {
    "p1": {"type": "real", "space": "linear", "range": (1e-5, 1)},
    "p2": {"type": "real", "space": "linear", "range": (1e-5, 1)},
    "p3": {"type": "real", "space": "linear", "range": (1e-5, 1)},
    "p4": {"type": "real", "space": "linear", "range": (1e-5, 1)},
    "p5": {"type": "real", "space": "linear", "range": (1e-5, 1)}
}

def input(op):
    global model
    global X_list
    global y_list
    global operator
    global max_margin_labels
    X_list = []
    y_list = []
    max_margin_labels = None
    operator = op
    print(operator)
    model = _get_model_name(operator)
    if platform.system() == "Windows":
        target_dir = None
        target_dir_parent = "../../data/models/model-operator/" + model
        for file in os.listdir(target_dir_parent):
            if file[-4:] != ".csv":
                if operator in file:
                    target_dir = os.path.join(target_dir_parent,file)
                    break
                
    else:
        target_dir = None
        target_dir_parent = "/home/falcon/data/models/model-operator/%s/" % (model)
        for file in os.listdir(target_dir_parent):
            if file[-4:] != ".csv":
                if operator in file:
                    target_dir = os.path.join(target_dir_parent,file)
                    break   
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            path = os.path.join(root, file)
            df = (pd.read_csv(path, header=None, index_col=False, engine="c", na_filter=False, true_values=["true"],
                              false_values=["false"]))
            df.columns = ["col1", "col2", "col3"]
            label = df.columns[-1]  # Assume last col is target
            target = df.pop(label).values
            # values = robust_standardize(df.values, q_level=0.86)
            # target = robust_standardize(target, q_level=0.86)
            values = np.array(df.values)
            target = np.array(target)
            X_list.append(values)
            y_list.append(target)

def _get_model_name(operator):
    if operator == 'add':
        return 'bert-large'
    elif operator == 'batch_norm':
        return 'densenet-201'
    elif operator == 'concat':
        return 'gru'
    elif operator == 'conv1d':
        return 'tacotron2'
    elif operator == 'conv2d':
        return 'inception-v2'
    elif operator == 'dense':
        return 'gru'
    elif operator == 'multiply':
        return 'gru'
    elif operator == 'relu':
        return 'densenet-201'
    elif operator == 'sigmoid':
        return 'gru'
    elif operator == 'split':
        return 'gru'
    elif operator == 'strided_slice':
        return 'gru'
    elif operator == 'tanh':
        return 'tacotron2'
    elif operator == 'transpose':
        return 'transformer'
    else:
        return None

class MCTSBOSearcher():
    primary_import = 'scikit-learn'

    def __init__(self, **kwargs):

        self.X_init = None
        self.batch_size = None
        self.mctsbo = None
        self.split_used = 0
        self.node = None
        self.target = None
        self.space_x = JointSpace(api_config)
        self.bounds = self.space_x.get_bounds()
        self.bounds = self.space_x.get_bounds()
        self.lb, self.ub = self.bounds[:, 0], self.bounds[:, 1]
        self.X = []
        self.y = []

        self.config = self._read_config()
#         print('config:', self.config)
        self.is_init_batch = False
        self.init_batches = []
        self.n_init = 1

    def _read_config(self):
        return {'turbo_training_steps': 100, 'turbo_length_retries': 3, 'turbo_length_init_method': 'default',
                'experimental_design': 'lhs_classic_ratio', 'n_init_points': 5, 'max_tree_depth': 9,
                'kmeans_resplits': 10,
                'split_model': {'type': 'SVC', 'args': {'kernel': 'poly', 'gamma': 'scale', 'C': 1000023}},
                'reset_no_improvement': 10, 'reset_split_after': 10,
                'mctsbo': {'budget': 128, 'use_cylinder': 0, 'use_pull': 0, 'use_lcb': 0, 'kappa': 2.0, 'use_decay': 1,
                          'decay_alpha': 0.49937937259674076, 'decay_threshold': 0.5, 'length_min': 1e-06,
                          'length_max': 2.0, 'length_init': 0.8, 'length_multiplier': 2.0}}

    def _get_split_model(self, X, kmeans_labels):
        global max_margin_labels
        split_model_config = self.config['split_model']
        model_type = split_model_config['type']
        args = split_model_config['args']
        if model_type == 'SVC':
            split_model = SVC(**args, max_iter=10 ** 7)
        elif model_type == 'KNeighborsClassifier':
            split_model = KNeighborsClassifier(**args)
        else:
            raise ValueError(f'Unknown split model type in the config: {model_type}.')
        # print("X:", X)
        # print("labels:", kmeans_labels)
        split_model.fit(X, kmeans_labels)
        split_model_predictions = max_margin_labels
        split_model_matches = np.sum(split_model_predictions == kmeans_labels)
        split_model_mismatches = np.sum(split_model_predictions != kmeans_labels)
        print('Labels for the split model:', kmeans_labels)
        print('Predictions of the split model:', split_model_predictions)
        print(f'Split model matches {split_model_matches} and mismatches {split_model_mismatches}')
        return split_model

    def _find_split(self, X, y) -> Optional:
        global max_margin_labels
        max_margin = None
        max_margin_labels = None
        km = None
        for _ in range(self.config['kmeans_resplits']):
            kmeans = KMeans(n_clusters=2).fit(y)
            kmeans_labels = kmeans.labels_

            if np.count_nonzero(kmeans_labels == 1) > 0 and np.count_nonzero(kmeans_labels == 0) > 0:
                if np.mean(y[kmeans_labels == 1]) < np.mean(y[kmeans_labels == 0]):
                    # Reverse labels if the entries with 1s have a higher mean error, since 1s go to the left branch.
                    kmeans_labels = 1 - kmeans_labels
                else:
                    km = kmeans
                margin = -(np.mean(y[kmeans_labels == 1]) - np.mean(y[kmeans_labels == 0]))
                if max_margin is None or margin > max_margin:
                    max_margin = margin
                    max_margin_labels = kmeans_labels
        print('Search areas, 1 means "good" search points, 0 means "bad" search points:', kmeans_labels)
        if max_margin_labels is None:
            return None
        else:
#             print(max_margin_labels)
            return km
            # return self._get_split_model(X, max_margin_labels), km

    def _build_tree(self, X, y, depth=0, index=0):
        # print("X:", X)
        # print('len(X) in _build_tree is', len(X))

        if index % 4 == 0:
            lb = np.array(self.lb)[[0, -1]]
            ub = np.array(self.ub)[[0, -1]]
        elif index % 4 == 1:
            lb = np.array(self.lb)[[1, -1]]
            ub = np.array(self.ub)[[1, -1]]
        elif index % 4 == 2:
            lb = np.array(self.lb)[[1, -2]]
            ub = np.array(self.ub)[[1, -2]]
        elif index % 4 == 3:
            lb = np.array(self.lb)[[1, 2]]
            ub = np.array(self.ub)[[1, 2]]
        X = deepcopy(X)
        y = deepcopy(y)

        print('Building the tree/branch for the %s operator.' % operator)
        print('Configuration candidates in current tree/branch is', len(X))
        if depth == self.config['max_tree_depth']:
            return []
        split = self._find_split(X, y.reshape(-1, 1))
        # _, split = self._find_split(X, y.reshape(-1, 1))
        if split is None:
            return []
        # in_region_points = split.predict(X)
        in_region_points = max_margin_labels
        left_subtree_size = np.count_nonzero(in_region_points == 1)
        right_subtree_size = np.count_nonzero(in_region_points == 0)
        print(f'{len(X)} configuration candidates would be split {left_subtree_size}/{right_subtree_size} of "good/bad" points.')

        idx = (in_region_points == 1)

        # X = X_list[(index + 1) % 4][idx[0:len(X_list[(index + 1) % 4])]]
        # y = y_list[(index + 1) % 4][idx[0:len(y_list[(index + 1) % 4])]]
        X_list[index % 4] = X_list[index % 4][idx]
        print("result", (X_list[index % 4]))
        if left_subtree_size <= self.n_init:
            return []

        if index > 2:
            return [split]

        # splits = self._build_tree(X[idx], y[idx], depth + 1)
        # print("X_list:", X_list)
        splits = self._build_tree(X_list[(index + 1) % 4], y_list[(index + 1) % 4], depth + 1, index + 1)
        return [split] + splits

    def _get_in_node_region(self, points, splits):
        # for i in range(0, len(splits)):
        #     if i == 0:
        #         split_in_region, splits[i] = splits[i].predict(np.array(points)[:, [0, -1]])
        #     elif i == 1:
        #         split_in_region = splits[i].predict(np.array(points)[:, [1, -1]])
        #     elif i == 2:
        #         split_in_region = splits[i].predict(np.array(points)[:, [1, -2]])
        #     elif i == 3:
        #         split_in_region = splits[i].predict(np.array(points)[:, [1, 2]])
        # print(split_in_region)
        in_region = np.ones(len(points))
        for split in splits:
            split_in_region = split.predict(points)
            # print("split_in_region:", split_in_region)
            in_region *= split_in_region
        return in_region
        #     in_region *= split_in_region
        # return in_region

    def build_tree_for_op(self):

        X = np.array(X_list[0])
        y = np.array(y_list[0])
        if not self.node:
            self.split_used = 0
            self.node = self._build_tree(X, y)
            self._fuse_tree()


    def _fuse_tree(self):

        batchsize_set = set()
        cpus_set = set()
        gpumem_set = set()
        gpupower_set = set()
        gputype_set = set()
        for i in range(0, len(X_list)):
            for line in X_list[i]:
                if i == 0:
                    batchsize_set.add(line[0])
                    gputype_set.add(line[1])
                elif i == 1:
                    cpus_set.add(line[0])
                    gputype_set.add(line[1])
                elif i == 2:
                    cpus_set.add(line[0])
                    gpupower_set.add(line[1])
                elif i == 3:
                    cpus_set.add(line[0])
                    gpumem_set.add(line[1])
        if len(batchsize_set) == 0:
            batchsize_set = {4, 8, 16, 32, 64, 128}
        if len(cpus_set) == 0:
            cpus_set = {1, 2, 3, 4, 5}
        if len(gpumem_set) == 0:
            gpumem_set = {0.8, 1.2, 1.6, 2.4}
        if len(gpupower_set) == 0:
            gpupower_set = {50, 75, 100}
        if len(gputype_set) == 0:
            gputype_set = {1, 2, 3, 4}
        for batchsize in batchsize_set:
            for cpus in cpus_set:
                for gpumem in gpumem_set:
                    for gpupower in gpupower_set:
                        for gputype in gputype_set:
                            self.X.append(standardize.standardize([batchsize, cpus, gpumem, gpupower, gputype]))
                            sum_y = np.array(
                                [float(query.query_op_data1(model, operator, 0, float(batchsize), float(gputype))[-1]),
                                 float(query.query_op_data1(model, operator, 1, float(cpus), float(gputype))[-1]),
                                 float(query.query_op_data1(model, operator, 2, float(cpus), float(gpupower))[-1]),
                                 float(query.query_op_data1(model, operator, 3, float(cpus), float(gpumem))[-1])])
                            # print(temp_y)
                            self.y.append(np.mean(sum_y))
                            # self.y.append(float(query.query_model_data(model, batchsize, cpus, gpumem, gpupower, gputype)[-1]))
#         print("X:", self.X)
#         print("y:", self.y)
        return self.X, self.y

def main():
    if len(sys.argv) == 2:
        op = sys.argv[1]
        input(op)
        opt = MCTSBOSearcher()
        opt.build_tree_for_op()
    else:
        for op in all_o:
            input(op)
            opt = MCTSBOSearcher()
            opt.build_tree_for_op()
                    
if __name__ == '__main__':
#     print(query_model_data('bert-large', 64, 3, 1.2, 70, 3))
#     query_model_data('densenet')
    main()
#     for model in all_m:
#         analyze_model(model)
#     print(u'\u2588\u2588'+' '+u'\u2588\u2588'+' '+u'\u2588\u2588'+' '+u'\u2588\u2588' + '  43%')
