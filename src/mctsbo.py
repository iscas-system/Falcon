import json
import os
import platform
import sys
import csv
from time import sleep, time
import pandas as pd
import numpy as np
from torch import manual_seed
from bayesmark.np_util import argmin_2d, random

from bayesmark.constants import ITER
import bayesmark.random_search as rs
from bayesmark.experiment import logger

from bayesmark.space import JointSpace
from bayesmark.stats import robust_standardize
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from utils import sampler
from utils.turbo1 import TuRBO1
from turbo.utils import from_unit_cube, latin_hypercube, to_unit_cube
from copy import deepcopy
from typing import Optional
from utils.util import copula_standardize
from utils import query, standardize, devision
from utils.logger import set_logger

if platform.system() == "Windows":
    LOG = "D:/falcon.log"
else:
    LOG = "/var/log/falctl.log"
logger = set_logger(os.path.basename(__file__), LOG)

api_config = {
    "p1": {"type": "real", "space": "linear", "range": (1e-5, 1)},
    "p2": {"type": "real", "space": "linear", "range": (1e-5, 1)},
    "p3": {"type": "real", "space": "linear", "range": (1e-5, 1)},
    "p4": {"type": "real", "space": "linear", "range": (1e-5, 1)},
    "p5": {"type": "real", "space": "linear", "range": (1e-5, 1)}
}

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

def input(arg):
    global model
    global X_list
    global y_list
    global operator
    global max_margin_labels
    global islast
    global iteration
    X_list = []
    y_list = []
    max_margin_labels = None
    if arg[0] not in all_m:
        print("Model should be contained in target sets.")
        logger.error("Model should be contained in target sets.")
        sys.exit(0)
    model = arg[0]
    operator = arg[1]
    islast = arg[2]
    iteration = arg[3]
    if platform.system() == "Windows":
        target_dir = None
        target_dir_parent = "..\\data\\models\\model-operator\\" + model
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


    # X_list = standardize.standardize(X_list)



# X_list = np.array(X_list)
# y_list = np.array(y_list)

try:
    import open3d

    DEBUG = True
except ImportError as _:
    DEBUG = False


def _add_pcd(pcds, points, color):
    if len(points) == 0:
        return
    if points.shape[1] == 2:
        extended_points = np.zeros((len(points), 3))
        extended_points[:, :2] = points[:, :]
        points = extended_points
    elif points.shape[1] != 3:
        raise ValueError('The points for the DEBUG should either be 2D or 3D.')
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
    pcds.append(pcd)

def fix_optimizer_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        manual_seed(seed)

class MCTSBOSearcher():
    primary_import = 'scikit-learn'

    def __init__(self, **kwargs):

        self.X_init = None
        self.batch_size = None
        self.mctsbo = None
        self.split_used = 0
        self.node = None
        self.target = None
        self.best_values = []
        self.space_x = JointSpace(api_config)
        self.bounds = self.space_x.get_bounds()
        self.lb, self.ub = self.bounds[:, 0], self.bounds[:, 1]

        self.dim = len(self.bounds)
        self.X = []
        self.y = []

        self.config = self._read_config()
#         print('config:', self.config)
        optimizer_seed = self.config.get('optimizer_seed')
        fix_optimizer_seed(optimizer_seed)
        self.sampler_seed = self.config.get('sampler_seed')
        sampler.fix_sampler_seed(self.sampler_seed)

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

    def _init(self, n_suggestions):
        self.batch_size = n_suggestions
        n_init_points = self.config['n_init_points']
        if n_init_points == -1:
            # Special value to use the default 2*D+1 number.
            n_init_points = 2 * self.dim + 1

        self.n_init = max(self.batch_size, n_init_points)

        # exp_design = self.config['experimental_design']
        # if exp_design == 'latin_hypercube':
        #     X_init = latin_hypercube(self.n_init, self.dim)
        # elif exp_design == 'halton':
        #     halton_sampler = sampler.Sampler(method='halton', api_config=self.api_config, n_points=self.n_init)
        #     X_init = halton_sampler.generate(random_state=self.sampler_seed)
        #     X_init = self.space_x.warp(X_init)
        #     X_init = to_unit_cube(X_init, self.lb, self.ub)
        # elif exp_design == 'lhs_classic_ratio':
        #     lhs_sampler = sampler.Sampler(
        #         method='lhs',
        #         api_config=api_config,
        #         n_points=self.n_init,
        #         generator_kwargs={'lhs_type': 'classic', 'criterion': 'ratio'})
        #     X_init = lhs_sampler.generate(random_state=self.sampler_seed)
        #     X_init = self.space_x.warp(X_init)
        #     X_init = to_unit_cube(X_init, self.lb, self.ub)
        # else:
        #     raise ValueError(f'Unknown experimental design: {exp_design}.')
        X_init = self._suggest(n_suggestions)
        self.X_init = X_init
#         if DEBUG:
#             print(f'Initialized the method with {self.n_init} points by {exp_design}:')
#             print(X_init)

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
                if DEBUG:
                    print('MARGIN is', margin, np.count_nonzero(kmeans_labels == 1),
                          np.count_nonzero(kmeans_labels == 0))
                if max_margin is None or margin > max_margin:
                    max_margin = margin
                    max_margin_labels = kmeans_labels
        print('Search areas, 1 means "good" search points, 0 means "bad" search points: %s' % kmeans_labels)
        logger.debug('Search areas, 1 means "good" search points, 0 means "bad" search points: %s' % kmeans_labels)
        if DEBUG:
            print('MAX MARGIN is', max_margin)
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

        print('Building the tree/branch for MCTS-BO searcher.')
        logger.debug('Building the tree/branch for MCTS-BO searcher.')
        print('Configuration candidates in current tree/branch is %s' % len(X))
        logger.debug('Configuration candidates in current tree/branch is %s' % len(X))
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
        logger.debug(f'{len(X)} configuration candidates would be split {left_subtree_size}/{right_subtree_size} of "good/bad" points.')

        idx = (in_region_points == 1)

        # X = X_list[(index + 1) % 4][idx[0:len(X_list[(index + 1) % 4])]]
        # y = y_list[(index + 1) % 4][idx[0:len(y_list[(index + 1) % 4])]]
        X_list[index % 4] = X_list[index % 4][idx]
        print("result %s" % (X_list[index % 4]))
        logger.debug("result %s" % (X_list[index % 4]))
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

    def _suggest(self, n_suggestions):

        # X = to_unit_cube(deepcopy(X_list[0]), self.lb, self.ub)
        # print("self.x:", self.X)

        # X = to_unit_cube(deepcopy(X_list[0]), self.lb, self.ub)
        # y = deepcopy(y_list[0])
        X = np.array(X_list[0])
        y = np.array(y_list[0])
        if not self.node:
            self.split_used = 0
            self.node = self._build_tree(X, y)
            # self.node = self._build_tree(X_list, y_list)
            used_budget = len(y)
            # idx = (self._get_in_node_region(X, self.node) == 1)
            # print("idx:", idx)
            self._fuse_tree()
            # X = X[idx]
            # y = y[idx]
#             print(f'Rebuilt the tree of depth {len(self.node)}')
            model_config = self.config['mctsbo']
            # print('CONFIG!!!!!', model_config)
            # lb = []
            # ub = []
            # for i in range(0, len(y)):
            #     lb.append(0)
            #     ub.append(1)

            self.mctsbo = TuRBO1(
                f=None,
                lb=self.bounds[:, 0],
                ub=self.bounds[:, 1],
                # lb=np.array(lb),
                # ub=np.array(ub),
                n_init=len(self.X),
                max_evals=np.iinfo(np.int32).max,
                batch_size=n_suggestions,
                verbose=False,
                use_cylinder=model_config['use_cylinder'],
                budget=model_config['budget'],
                use_decay=model_config['use_decay'],
                decay_threshold=model_config['decay_threshold'],
                decay_alpha=model_config['decay_alpha'],
                use_pull=model_config['use_pull'],
                use_lcb=model_config['use_lcb'],
                kappa=model_config['kappa'],
                length_min=model_config['length_min'],
                length_max=model_config['length_max'],
                length_init=model_config['length_init'],
                length_multiplier=model_config['length_multiplier'],
                used_budget=used_budget
            )
            self.mctsbo._X = np.array(self.X, copy=True)
            self.mctsbo._fX = np.array(self.y, copy=True)
            self.mctsbo.X = np.array(self.X, copy=True)
            self.mctsbo.fX = np.array(self.y, copy=True)
            print('Initializing the MCTS-BO search algorithm, please wait...')
            logger.debug('Initializing the MCTS-BO search algorithm, please wait...')

        else:
            # idx = (self._get_in_node_region(self.X, self.node) == 1)
            idx = (self._get_in_node_region(self.y, self.node) == 1)
            # print("idx:", idx)
            self.X = self.X[idx]
            self.y = self.y[idx]
        self.split_used += 1
        # print("X:", self.X)
        # print("y:", self.y)
        length_init_method = self.config['turbo_length_init_method']
        if length_init_method == 'default':
            length = self.mctsbo.length
        elif length_init_method == 'length_init':
            length = self.mctsbo.length_init
        elif length_init_method == 'length_max':
            length = self.mctsbo.length_max
        elif length_init_method == 'infinity':
            length = np.iinfo(np.int32).max
        else:
            raise ValueError(f'Unknown init method for mctsbo\'s length: {length_init_method}.')
        length_reties = self.config['turbo_length_retries']

        for retry in range(length_reties):
            XX = np.array(self.X)
            # print("XX:", XX)
            yy = np.array(1 - np.array(self.y)).reshape(-1)
            lb = np.array([min(XX[:, 0]), min(XX[:, 1]), min(XX[:, 2]), min(XX[:, 3]), min(XX[:, 4])])
            ub = np.array([max(XX[:, 0]), max(XX[:, 1]), max(XX[:, 2]), max(XX[:, 3]), max(XX[:, 4])])
            # print("lb:", lb)
            # print("ub:", ub)
            # print("yy", yy.ndim)
            X_cand, y_cand, _ = self.mctsbo._create_candidates(
                XX, yy, lb, ub, length=length, n_training_steps=self.config['turbo_training_steps'], hypers={})
            # print("X_cand:", X_cand)
            # print("y_cand:", y_cand)
            # in_region_predictions = self._get_in_node_region(X_cand, self.node)
            in_region_predictions = self._get_in_node_region(y_cand, self.node)
            in_region_idx = in_region_predictions == 1
            if DEBUG:
                print(f'In region: {np.sum(in_region_idx)} out of {len(X_cand)}')
            if np.sum(in_region_idx) >= n_suggestions:
                X_cand, y_cand = X_cand[in_region_idx], y_cand[in_region_idx]
                self.mctsbo.f_var = self.mctsbo.f_var[in_region_idx]
                if DEBUG:
                    print('Found a suitable set of candidates.')
                break
            else:
                length /= 2
                if DEBUG:
                    print(f'Retrying {retry + 1}/{length_reties} time')
        # print("X_cand:", X_cand)
        # print("y_cand:", y_cand)
        X_cand = self.mctsbo._select_candidates(X_cand, y_cand)[:n_suggestions, :]
        return X_cand

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
                                [float(query.query_op_data(model, operator, 0, float(batchsize), float(gputype))[-1]),
                                 float(query.query_op_data(model, operator, 1, float(cpus), float(gputype))[-1]),
                                 float(query.query_op_data(model, operator, 2, float(cpus), float(gpupower))[-1]),
                                 float(query.query_op_data(model, operator, 3, float(cpus), float(gpumem))[-1])])
                            # print(temp_y)
                            self.y.append(np.mean(sum_y))
                            # self.y.append(float(query.query_model_data(model, batchsize, cpus, gpumem, gpupower, gputype)[-1]))
#         print("X:", self.X)
#         print("y:", self.y)
        return self.X, self.y

    def _jump_area(self, jump):
        region = np.array(devision.devide_region(model, operator)[jump])
        # print("region:", region)
        x = np.array([i[0] for i in region])
        y = np.array([i[1] for i in region])
        return x, y.reshape(-1, 1)

    def suggest(self, n_suggestions=5):
        X_suggestions = np.zeros((n_suggestions, self.dim))
        # Initialize the design if it is the first call

        if self.X_init is None:
            self._init(n_suggestions)
            if self.init_batches:
                print('REUSING INITIALIZATION:')
                for X, Y in self.init_batches:
                    print('Re-observing a batch!')
                    self.observe(X, Y)
                self.X_init = []

        # Pick from the experimental design
        # print("X_init:", len(self.X_init))
        # print("n_suggestions:", n_suggestions)
        n_init = min(len(self.X_init), n_suggestions)
        if n_init > 0:
            X_suggestions[:n_init] = self.X_init[:n_init]
            self.X_init = self.X_init[n_init:]
            self.is_init_batch = True
        else:
            self.is_init_batch = False

        # print("suggest:", n_suggestions)
        # print("init:", n_init)
        # Pick from the model based on the already received observations
        n_suggest = n_suggestions - n_init
        if n_suggest > 0:
            # n_suggest = n_suggestions

            X_cand = self._suggest(n_suggest)
            # print("X_cand:", X_cand)
            X_suggestions[-n_suggest:] = X_cand

        # Map into the continuous space with the api bounds and unwarp the suggestions
        X_min_bound = -1
        # X_max_bound = 1.0
        X_max_bound = 2
        X_suggestions_min = X_suggestions.min()
        X_suggestions_max = X_suggestions.max()
        # if X_suggestions_min < X_min_bound or X_suggestions_max > X_max_bound:
        #     print(f'Some suggestions are out of the bounds in suggest(): {X_suggestions_min}, {X_suggestions_max}')
        #     print('Clipping everything...')
        #     X_suggestions = np.clip(X_suggestions, X_min_bound, X_max_bound)
        # X_suggestions = from_unit_cube(X_suggestions, self.lb, self.ub)
        # X_suggestions = self.space_x.unwarp(X_suggestions)

        return X_suggestions

    def observe(self, X_observed, Y_observed):
        if self.is_init_batch:
            self.init_batches.append([X_observed, Y_observed])
        X, Y = [], []
        for x, y in zip(X_observed, Y_observed):
            if np.isfinite(y):
                X.append(x)
                Y.append(y)
            else:
                # Ignore for now; could potentially substitute with an upper bound.
                continue
        if not X:
            return
        # X, Y = self.space_x.warp(X), np.array(Y)[:, None]
        Y = np.array(Y)[:, None]
        # print("self.X:", self.X)
        # print("X:", X)

        Y = np.array(Y_observed).reshape(-1, 1)

        self.X = np.vstack((np.array(self.X), np.array(standardize.standardize(deepcopy(X[0])))))
        # self.y = np.vstack((np.array(self.y), deepcopy(Y)))
        # print("self.y:", np.array(self.y).reshape(-1, 1))
        # print("Y:", np.array(Y).reshape(-1, 1))
        self.y = np.vstack((np.array(self.y).reshape(-1, 1), deepcopy(Y).reshape(-1, 1)))
        self.best_values.append(Y.min())

        if self.mctsbo:
            if len(self.mctsbo._X) >= self.mctsbo.n_init:
                self.mctsbo._adjust_length(Y)
#             print('TURBO length:', self.mctsbo.length)
            self.mctsbo._X = np.vstack((self.mctsbo._X, deepcopy(X)))
            # print("turbo_fx:", self.mctsbo._fX)

            self.mctsbo._fX = np.vstack((self.mctsbo._fX.reshape(-1, 1), deepcopy(Y).reshape(-1, 1)))
            # self.mctsbo._fX = np.vstack((self.mctsbo._fX, deepcopy(Y)))
            self.mctsbo.X = np.vstack((self.mctsbo.X, deepcopy(X)))
            self.mctsbo.fX = np.vstack((self.mctsbo.fX.reshape(-1, 1), deepcopy(Y).reshape(-1, 1)))
            # self.mctsbo.fX = np.vstack((self.mctsbo.fX, deepcopy(Y)))

        N = self.config['reset_no_improvement']
        if len(self.best_values) > N and np.min(self.best_values[:-N]) <= np.min(self.best_values[-N:]):
            print('########## RESETTING COMPLETELY! ##########')
            self.X = np.zeros((0, self.dim))
            self.y = np.zeros((0, 1))
            self.best_values = []
            self.X_init = None
            self.node = None
            self.mctsbo = None
            self.split_used = 0

        if self.split_used >= self.config['reset_split_after']:
            print('########## REBUILDING THE SPLIT! ##########')
            self.node = None
            self.mctsbo = None
            self.split_used = 0

    def run_study(self, n_calls, n_suggestions, n_obj=1, callback=None, api_config=api_config):
        """Run a study for a single optimizer on a single test problem.

        This function can be used for benchmarking on general stateless objectives (not just `sklearn`).

        Parameters
        ----------
        optimizer : :class:`.abstract_optimizer.AbstractOptimizer`
            Instance of one of the wrapper optimizers.
        test_problem : :class:`.sklearn_funcs.TestFunction`
            Instance of test function to attempt to minimize.
        n_calls : int
            How many iterations of minimization to run.
        n_suggestions : int
            How many parallel evaluation we run each iteration. Must be ``>= 1``.
        n_obj : int
            Number of different objectives measured, only objective 0 is seen by optimizer. Must be ``>= 1``.
        callback : callable
            Optional callback taking the current best function evaluation, and the number of iterations finished. Takes
            array of shape `(n_obj,)`.

        Returns
        -------
        function_evals : :class:`numpy:numpy.ndarray` of shape (n_calls, n_suggestions, n_obj)
            Value of objective for each evaluation.
        timing_evals : (:class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`)
            Tuple of 3 timing results: ``(suggest_time, eval_time, observe_time)`` with shapes ``(n_calls,)``,
            ``(n_calls, n_suggestions)``, and ``(n_calls,)``. These are the time to make each suggestion, the time for each
            evaluation of the objective function, and the time to make an observe call.
        suggest_log : list(list(dict(str, object)))
            Log of the suggestions corresponding to the `function_evals`.
        """
        assert n_suggestions >= 1, "batch size must be at least 1"
        assert n_obj >= 1, "Must be at least one objective"
        global iteration
        
        if platform.system() == "Windows":
            csv_path = os.path.join(current_dir,"../data/experiments/falcon-%s.csv" % model)
        else:
            csv_path = "/home/falcon/data/experiments/falcon-%s.csv" % model
#         # space_for_validate = JointSpace(test_problem.get_api_config())
#         space_for_validate = JointSpace(api_config)
        if callback is not None:
            # First do initial log at inf score, in case we don't even get to first eval before crash/job timeout
            callback(np.full((n_obj,), np.inf, dtype=float), 0)

        suggest_time = np.zeros(n_calls)
        observe_time = np.zeros(n_calls)
        eval_time = np.zeros((n_calls, n_suggestions))
        evals = []
        function_evals = np.zeros((n_calls, n_suggestions, n_obj))
        suggest_log = []
        stop_flag = False
        for ii in range(n_calls):
            # print("selfX:", self.X)
            if ii == 2:
                self.X, self.y = self._jump_area(0)

            # tt = time()
            try:
                next_points = self.suggest(n_suggestions)

            except Exception as e:
                logger.warning("Failure in optimizer suggest. Falling back to random search.")
                logger.exception(e, exc_info=True)
                print(json.dumps({"optimizer_suggest_exception": {ITER: ii}}))
                api_config = api_config
                _next_points = rs.suggest_dict([], [], api_config, n_suggestions=n_suggestions)
                next_points = []
                for _next_point in _next_points:
                    next_points.append(
                        [_next_point['p1'], _next_point['p2'], _next_point['p3'], _next_point['p4'], _next_point['p5']])

            logger.info("suggestion time taken %f iter %d next_points %s" % (suggest_time[ii], ii, str(next_points)))
            assert len(next_points) == n_suggestions, "invalid number of suggestions provided by the optimizer"

            for jj, next_point in enumerate(next_points):
                # tt = time()
                try:
                    next_point = standardize.unstandardize(next_point)
                    f_current_eval = float(query.query_model_data(model, int(next_point[0]), int(next_point[1]),
                                                                  next_point[2], int(next_point[3]),
                                                                  int(next_point[4]))[-1])

                    next_point_output = {'batchsize': int(next_point[0]), 'cpus': int(next_point[1]), 'gpumem':next_point[2],\
                                  'gpupower': int(next_point[3]), 'gputype': all_gputype[int(next_point[4])-1]}
                    iteration += 1
                    next_point_write_to_file = [model, iteration, int(next_point[0]), int(next_point[1]),
                                                next_point[2], int(next_point[3]), 
                                                all_gputype[int(next_point[4])-1], f_current_eval]
                    with open(csv_path, 'a', newline='') as fr:
                        writer = csv.writer(fr)
                        writer.writerow(next_point_write_to_file) 
                    print("Next recommended configuration: %s"  % next_point_output)
                    logger.debug("Next recommended configuration: %s" % next_point_output)
#                     print("next iteration:", next_point_write_to_file)
                    print("Norm.(RPS/Budget): %s" % f_current_eval)
                    logger.debug("Norm.(RPS/Budget): %s" % f_current_eval)
                except Exception as e:
                    logger.warning("Failure in function eval. Setting to inf.")
                    logger.exception(e, exc_info=True)
                    f_current_eval = np.full((n_obj,), np.inf, dtype=float)

                suggest_log.append(next_points)
                evals.append(f_current_eval)
                function_evals[ii, jj, :] = f_current_eval

                if f_current_eval > 0.9:
                    stop_flag = True
                    break

            if stop_flag or (ii == 4 and not islast):
                return (max(evals), np.array(suggest_log[np.argmax(evals)]).reshape(-1), iteration)

            # Note: this could be inf in the event of a crash in f evaluation, the optimizer must be able to handle that.
            # Only objective 0 is seen by optimizer.
            eval_list = function_evals[ii, :, 0].tolist()

            if callback is not None:
                idx_ii, idx_jj = argmin_2d(function_evals[: ii + 1, :, 0])
                callback(function_evals[idx_ii, idx_jj, :], ii + 1)

            tt = time()
            try:
                # print("obX:", next_points)
                # print("self_x:", self.X)

                self.observe(next_points, eval_list)
            except Exception as e:
                logger.warning("Failure in optimizer observe. Ignoring these observations.")
                logger.exception(e, exc_info=True)
                print(json.dumps({"optimizer_observe_exception": {ITER: ii}}))
            observe_time[ii] = time() - tt

            logger.info(
                "observation time %f, current best %f at iter %d"
                % (observe_time[ii], np.min(function_evals[: ii + 1, :, 0]), ii)
            )
        return (max(evals), np.array(suggest_log[np.argmax(evals)]).reshape(-1), iteration)
    
def main(argv):
    input(argv)
    opt = MCTSBOSearcher()
    return opt.run_study(6, 1, 1)
