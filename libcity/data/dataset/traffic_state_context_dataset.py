# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 20:08:46 2022

@author: yatzhang
"""

import os
import numpy as np
import pandas as pd
from libcity.utils import ensure_dir
from torch.utils.data import Dataset
from libcity.data.dataset import TrafficStateDataset
from libcity.data.utils import context_data_padding, generate_dataloader_context


class Traffic_Context_Dataset(Dataset):
    def __init__(self, goal_data, sctx_data=None, tctx_data=None):
        self.goal_data = goal_data
        self.sctx_data = sctx_data
        self.tctx_data = tctx_data

    def __len__(self):
        return len(self.goal_data)

    # Style in LibCity: 2067-2-12-156-1， in which 2 refers to X and y respectively
    def __getitem__(self, idx):
        goal = self.goal_data[idx][0]
        target_goal = self.goal_data[idx][1]
        # sctx
        if self.sctx_data is not None and self.tctx_data is None:
            sctx = self.sctx_data[idx][0]
            target_sctx = self.sctx_data[idx][1]
            return goal, target_goal, sctx, target_sctx
        # tctx
        if self.sctx_data is None and self.tctx_data is not None :
            tctx = self.tctx_data[idx][0]
            target_tctx = self.tctx_data[idx][1]
            return goal, target_goal, tctx, target_tctx
        # sctx and tctx
        if self.sctx_data is not None and self.tctx_data is not None:
            sctx = self.sctx_data[idx][0]
            target_sctx = self.sctx_data[idx][1]
            tctx = self.tctx_data[idx][0]
            target_tctx = self.tctx_data[idx][1]
            return goal, target_goal, sctx, target_sctx, tctx, target_tctx


class TrafficStateContextDataset(TrafficStateDataset):
    def __init__(self, config):
        super().__init__(config)
        self.spatial_base = self.config.get('spatial_base', 'none')
        self.temporal_base = self.config.get('temporal_base', 'none')
        self.drop_tctx = self.config.get('drop_tctx', 'none')

        self.context_feature_name = None
        self.feature_sctx_dim = 0
        self.feature_tctx_dim = 0
        self.feature_total_dim = 0
        self.sctx_scaler = None
        self.tctx_scaler = None
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = None, None, None
        self.parameters_str = str(self.dataset) + '_' + str(self.input_window) + '_' + str(self.output_window) \
                              + '_' + str(self.train_rate) + '_' + str(self.eval_rate) + '_' + str(self.scaler_type)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/', 'point_based_{}.npz'.format(self.parameters_str))

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, properties(若干列)]
        """
        super()._load_geo()

    def _load_rel(self):
        """
        加载.rel文件，格式[rel_id, type, origin_id, destination_id, properties(若干列)]

        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        super()._load_rel()

    def _load_dyna(self, filename):
        """
        加载.dyna文件，格式[dyna_id, type, time, entity_id, properties(若干列)]
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组, 3d-array (len_time, num_nodes, feature_dim)
        """
        return super()._load_dyna_3d(filename)
    
    def _load_context_dyna(self, filename, context_base):
        """
        加载.dyna文件，格式[dyna_id, type, time, entity_id, properties(若干列)]
        本函数加载context数据，包括spatial context, temporal context

        Args:
            filename(str): 数据文件名，不包含后缀
            context_base(str): 联合filename生成context文件的文件名

        Returns:
            np.ndarray: 数据数组, 3d-array (len_time, num_nodes, feature_dim)
        """
        # 加载数据集
        context_name = filename + '_' + context_base
        self._logger.info("Loading file " + context_name + '.dyna')
        dynafile = pd.read_csv(self.data_path + context_name + '.dyna', low_memory=False)
        # remove some columns in tctx
        if context_base[-4:] == 'tctx':
            _drop_tctx = self.drop_tctx.split('-')
            for i_drop in _drop_tctx:
                if i_drop.strip() == 'jf':
                    dynafile = dynafile.drop(columns=[i_drop.strip()])
                elif i_drop.strip() == 'busnode':
                    dynafile = dynafile.drop(columns=['in_busnode'])
                    dynafile = dynafile.drop(columns=['out_busnode'])
                elif i_drop.strip() == 'weather':
                    dynafile = dynafile.drop(columns=['temperature'])
                    dynafile = dynafile.drop(columns=['rainfall'])
                    dynafile = dynafile.drop(columns=['windspeed'])
        dynafile = dynafile[dynafile.columns[2:]]  # 从time列开始所有列
        # 求时间序列
        self.timesolts = list(dynafile['time'][:int(dynafile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not dynafile['time'].isna().any():  # 时间没有空值
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # 转3-d数组
        feature_dim = len(dynafile.columns) - 2
        df = dynafile[dynafile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i:i + len_time].values)
        data = np.array(data, dtype=np.float)  # (len(self.geo_ids), len_time, feature_dim)
        data = data.swapaxes(0, 1)  # (len_time, len(self.geo_ids), feature_dim)
        self._logger.info("Loaded file " + context_name + '.dyna' + ', shape=' + str(data.shape))
        return data

    def _add_external_information(self, df, ext_data=None):
        """
        增加外部信息（一周中的星期几/day of week，一天中的某个时刻/time of day，外部数据）

        Args:
            df(np.ndarray): 交通状态数据多维数组, (len_time, num_nodes, feature_dim)
            ext_data(np.ndarray): 外部数据

        Returns:
            np.ndarray: 融合后的外部数据和交通状态数据, (len_time, num_nodes, feature_dim_plus)
        """
        return super()._add_external_information_3d(df, ext_data)

    def get_data(self): 
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        # load dataset
        x_train_goal, y_train_goal, x_val_goal, y_val_goal, x_test_goal, y_test_goal = None, None, None, None, None, None
        x_train_sctx, y_train_sctx, x_val_sctx, y_val_sctx, x_test_sctx, y_test_sctx = None, None, None, None, None, None
        x_train_tctx, y_train_tctx, x_val_tctx, y_val_tctx, x_test_tctx, y_test_tctx = None, None, None, None, None, None

        if self.data is None:
            self.data = {}
            # more variables are needed if more context data should be considered
            origin_cache_file_name = self.cache_file_name
            sctx_cache_file_name = origin_cache_file_name[:-4] + '_' + self.spatial_base + '.npz'
            tctx_cache_file_name = origin_cache_file_name[:-4] + '_' + self.temporal_base + '_drop-' + self.drop_tctx + '.npz'
            # goal
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train_goal, y_train_goal, x_val_goal, y_val_goal, x_test_goal, y_test_goal = self._load_cache_train_val_test()
            else:
                x_train_goal, y_train_goal, x_val_goal, y_val_goal, x_test_goal, y_test_goal = self._generate_train_val_test()
            # sctx
            if self.cache_dataset and os.path.exists(sctx_cache_file_name):
                if self.spatial_base[-4:] == 'sctx':
                    x_train_sctx, y_train_sctx, x_val_sctx, y_val_sctx, x_test_sctx, y_test_sctx = self._load_cache_train_val_test_context(self.spatial_base)
            else:
                if self.spatial_base[-4:] == 'sctx':
                    x_train_sctx, y_train_sctx, x_val_sctx, y_val_sctx, x_test_sctx, y_test_sctx = self._generate_train_val_test_context(self.spatial_base)
            # tctx
            if self.cache_dataset and os.path.exists(tctx_cache_file_name):
                if self.temporal_base[-4:] == 'tctx':
                    x_train_tctx, y_train_tctx, x_val_tctx, y_val_tctx, x_test_tctx, y_test_tctx = self._load_cache_train_val_test_context(self.temporal_base)
            else:
                if self.temporal_base[-4:] == 'tctx':
                    x_train_tctx, y_train_tctx, x_val_tctx, y_val_tctx, x_test_tctx, y_test_tctx = self._generate_train_val_test_context(self.temporal_base)

        # initialization for context data
        sctx_train_data, sctx_val_data, sctx_test_data = None, None, None
        tctx_train_data, tctx_val_data, tctx_test_data = None, None, None

        # Data scaling for all: goal and sctx
        goal_train_data, goal_val_data, goal_test_data = self._scaler_data(x_train_goal, y_train_goal, x_val_goal, y_val_goal, x_test_goal, y_test_goal, 'goal')
        if self.spatial_base[-4:] == 'sctx':
            sctx_train_data, sctx_val_data, sctx_test_data = self._scaler_data(x_train_sctx, y_train_sctx, x_val_sctx, y_val_sctx, x_test_sctx, y_test_sctx, 'sctx')
        # Data scaling by column: tctx
        if self.temporal_base[-4:] == 'tctx':
            tctx_train_data, tctx_val_data, tctx_test_data = self._scaler_data_by_column(x_train_tctx, y_train_tctx, x_val_tctx, y_val_tctx, x_test_tctx, y_test_tctx, 'tctx')

        # 对于若最后一个 batch 不满足 batch_size的情况， 是否进行补齐（使用最后一个元素反复填充补齐）
        if self.pad_with_last_sample:
            goal_train_data, goal_val_data, goal_test_data = context_data_padding(goal_train_data, goal_val_data, goal_test_data, self.batch_size)
            if self.spatial_base[-4:] == 'sctx':
                sctx_train_data, sctx_val_data, sctx_test_data = context_data_padding(sctx_train_data, sctx_val_data, sctx_test_data, self.batch_size)
            if self.temporal_base[-4:] == 'tctx':
                tctx_train_data, tctx_val_data, tctx_test_data = context_data_padding(tctx_train_data, tctx_val_data, tctx_test_data, self.batch_size)

        # context_feature_name
        # sctx
        if self.spatial_base[-4:] == 'sctx' and self.temporal_base[-4:] != 'tctx':
            self.context_feature_name = {'X_goal': 'float', 'y_goal': 'float',
                                         'X_sctx': 'float', 'y_sctx': 'float'}
        # tctx
        if self.spatial_base[-4:] != 'sctx' and self.temporal_base[-4:] == 'tctx':
            self.context_feature_name = {'X_goal': 'float', 'y_goal': 'float',
                                         'X_tctx': 'float', 'y_tctx': 'float'}
        # sctx and tctx
        if self.spatial_base[-4:] == 'sctx' and self.temporal_base[-4:] == 'tctx':
            self.context_feature_name = {'X_goal': 'float', 'y_goal': 'float',
                                         'X_sctx': 'float', 'y_sctx': 'float',
                                         'X_tctx': 'float', 'y_tctx': 'float'}

        # build training, validation, and test dataset
        train_dataset = Traffic_Context_Dataset(goal_train_data, sctx_train_data, tctx_train_data)
        val_dataset = Traffic_Context_Dataset(goal_val_data, sctx_val_data, tctx_val_data)
        test_dataset = Traffic_Context_Dataset(goal_test_data, sctx_test_data, tctx_test_data)

        # create dataloader for loading datasets into models
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader_context(train_dataset, val_dataset, test_dataset,
                                        self.context_feature_name, self.batch_size, self.num_workers,
                                        shuffle=False, pad_with_last_sample=self.pad_with_last_sample)

        # The format of train_dataloader: same as Traffic_Context_Dataset, i.e., 2067-4-12-156-1, but two differences
        # Dif 1: the 1st dim, i.e., the total length of 2067 is replaced by the batch_size, e.g., 8
        # Dif 2: the 2nd dim, i.e., 4 is replaced by the key of 'X_goal', 'y_goal', 'X_sctx', 'y_sctx'
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        self.feature_total_dim = self.feature_dim + self.feature_sctx_dim + self.feature_tctx_dim
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "ext_dim": self.ext_dim,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_total_dim,
                "feature_sctx_dim": self.feature_sctx_dim, "feature_tctx_dim": self.feature_tctx_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches}

    def _generate_train_val_test_context(self, context_base):
        """
        加载数据集，并划分训练集、测试集、验证集，并缓存数据集

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        x, y = self._generate_data_context(context_base)
        return self._split_train_val_test_context(x, y, context_base)
    
    def _generate_data_context(self, context_base):
        """
        加载数据文件(.dyna/.grid/.od/.gridod)和外部数据(.ext)，且将二者融合，以X，y的形式返回

        Returns:
            tuple: tuple contains:
                x(np.ndarray): 模型输入数据，(num_samples, input_length, ..., feature_dim) \n
                y(np.ndarray): 模型输出数据，(num_samples, output_length, ..., feature_dim)
        """
        # 处理多数据文件问题
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:  # str
            data_files = [self.data_files].copy()
        # 加载外部数据
        if self.load_external and os.path.exists(self.data_path + self.ext_file + '.ext'):  # 外部数据集
            ext_data = self._load_ext()
        else:
            ext_data = None
        x_list, y_list = [], []
        for filename in data_files:
            df = self._load_context_dyna(filename, context_base)  # (len_time, ..., feature_dim)
            if self.load_external:
                df = self._add_external_information(df, ext_data)
            x, y = self._generate_input_data(df)
            # x: (num_samples, input_length, ..., input_dim)
            # y: (num_samples, output_length, ..., output_dim)
            x_list.append(x)
            y_list.append(y)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        self._logger.info("Dataset created")
        self._logger.info("x shape: " + str(x.shape) + ", y shape: " + str(y.shape))
        return x, y
    
    def _split_train_val_test_context(self, x, y, context_base):          
        """
        划分训练集、测试集、验证集，并缓存数据集

        Args:
            x(np.ndarray): 输入数据 (num_samples, input_length, ..., feature_dim)
            y(np.ndarray): 输出数据 (num_samples, input_length, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = x.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train

        # train
        x_train, y_train = x[:num_train], y[:num_train]
        # val
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        # test
        x_test, y_test = x[-num_test:], y[-num_test:]
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
        
        origin_cache_file_name = self.cache_file_name
        context_cache_file_name = origin_cache_file_name[:-4] + '_' + context_base + '.npz'
        if context_base[-4:] == 'tctx':
            context_cache_file_name = origin_cache_file_name[:-4] + '_' + context_base + '_drop-' + self.drop_tctx + '.npz'
        
        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                context_cache_file_name,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                x_val=x_val,
                y_val=y_val,
            )
            self._logger.info('Saved at ' + context_cache_file_name)
        return x_train, y_train, x_val, y_val, x_test, y_test
    
    def _load_cache_train_val_test_context(self, context_base):
        """
        加载之前缓存好的训练集、测试集、验证集

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        origin_cache_file_name = self.cache_file_name
        context_cache_file_name = origin_cache_file_name[:-4] + '_' + context_base + '.npz'
        if context_base[-4:] == 'tctx':
            context_cache_file_name = origin_cache_file_name[:-4] + '_' + context_base + '_drop-' + self.drop_tctx + '.npz'
    
        self._logger.info('Loading ' + context_cache_file_name)
        cat_data = np.load(context_cache_file_name)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _scaler_data(self, x_train, y_train, x_val, y_val, x_test, y_test, context_base):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        # scale_dim = self.output_dim
        if context_base[-4:] == 'goal':
            self.feature_dim = x_train.shape[-1]
            scale_dim = self.output_dim
            self.scaler = self._get_scalar(self.scaler_type, x_train[..., :scale_dim], y_train[..., :scale_dim])
            x_train[..., :scale_dim] = self.scaler.transform(x_train[..., :scale_dim])
            y_train[..., :scale_dim] = self.scaler.transform(y_train[..., :scale_dim])
            x_val[..., :scale_dim] = self.scaler.transform(x_val[..., :scale_dim])
            y_val[..., :scale_dim] = self.scaler.transform(y_val[..., :scale_dim])
            x_test[..., :scale_dim] = self.scaler.transform(x_test[..., :scale_dim])
            y_test[..., :scale_dim] = self.scaler.transform(y_test[..., :scale_dim])
        elif context_base[-4:] == 'sctx':
            self.feature_sctx_dim = x_train.shape[-1]
            scale_dim = self.feature_sctx_dim
            self.sctx_scaler = self._get_scalar(self.scaler_type, x_train[..., :scale_dim], y_train[..., :scale_dim])
            x_train[..., :scale_dim] = self.sctx_scaler.transform(x_train[..., :scale_dim])
            y_train[..., :scale_dim] = self.sctx_scaler.transform(y_train[..., :scale_dim])
            x_val[..., :scale_dim] = self.sctx_scaler.transform(x_val[..., :scale_dim])
            y_val[..., :scale_dim] = self.sctx_scaler.transform(y_val[..., :scale_dim])
            x_test[..., :scale_dim] = self.sctx_scaler.transform(x_test[..., :scale_dim])
            y_test[..., :scale_dim] = self.sctx_scaler.transform(y_test[..., :scale_dim])
        elif context_base[-4:] == 'tctx':
            self.feature_tctx_dim = x_train.shape[-1]
            scale_dim = self.feature_tctx_dim
            self.tctx_scaler = self._get_scalar(self.scaler_type, x_train[..., :scale_dim], y_train[..., :scale_dim])
            x_train[..., :scale_dim] = self.tctx_scaler.transform(x_train[..., :scale_dim])
            y_train[..., :scale_dim] = self.tctx_scaler.transform(y_train[..., :scale_dim])
            x_val[..., :scale_dim] = self.tctx_scaler.transform(x_val[..., :scale_dim])
            y_val[..., :scale_dim] = self.tctx_scaler.transform(y_val[..., :scale_dim])
            x_test[..., :scale_dim] = self.tctx_scaler.transform(x_test[..., :scale_dim])
            y_test[..., :scale_dim] = self.tctx_scaler.transform(y_test[..., :scale_dim])
        
        # 把训练集的X和y聚合在一起成为list，测试集验证集同理
        # x_train/y_train: (num_samples, input_length, ..., feature_dim)
        # train_data(list): train_data[i]是一个元组，由x_train[i]和y_train[i]组成
        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        return train_data, eval_data, test_data

    def _scaler_data_by_column(self, x_train, y_train, x_val, y_val, x_test, y_test, context_base):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        # scale_dim = self.output_dim
        if context_base[-4:] == 'goal':
            self.feature_dim = x_train.shape[-1]
            scale_dim = self.output_dim
            for i_dim in range(0, scale_dim):
                self.scaler = self._get_scalar(self.scaler_type, x_train[..., i_dim:i_dim + 1], y_train[..., i_dim:i_dim + 1])
                x_train[..., i_dim:i_dim + 1] = self.scaler.transform(x_train[..., i_dim:i_dim + 1])
                y_train[..., i_dim:i_dim + 1] = self.scaler.transform(y_train[..., i_dim:i_dim + 1])
                x_val[..., i_dim:i_dim + 1] = self.scaler.transform(x_val[..., i_dim:i_dim + 1])
                y_val[..., i_dim:i_dim + 1] = self.scaler.transform(y_val[..., i_dim:i_dim + 1])
                x_test[..., i_dim:i_dim + 1] = self.scaler.transform(x_test[..., i_dim:i_dim + 1])
                y_test[..., i_dim:i_dim + 1] = self.scaler.transform(y_test[..., i_dim:i_dim + 1])
        elif context_base[-4:] == 'sctx':
            self.feature_sctx_dim = x_train.shape[-1]
            scale_dim = self.feature_sctx_dim
            for i_dim in range(0, scale_dim):
                self.sctx_scaler = self._get_scalar(self.scaler_type, x_train[..., i_dim:i_dim + 1], y_train[..., i_dim:i_dim + 1])
                x_train[..., i_dim:i_dim + 1] = self.sctx_scaler.transform(x_train[..., i_dim:i_dim + 1])
                y_train[..., i_dim:i_dim + 1] = self.sctx_scaler.transform(y_train[..., i_dim:i_dim + 1])
                x_val[..., i_dim:i_dim + 1] = self.sctx_scaler.transform(x_val[..., i_dim:i_dim + 1])
                y_val[..., i_dim:i_dim + 1] = self.sctx_scaler.transform(y_val[..., i_dim:i_dim + 1])
                x_test[..., i_dim:i_dim + 1] = self.sctx_scaler.transform(x_test[..., i_dim:i_dim + 1])
                y_test[..., i_dim:i_dim + 1] = self.sctx_scaler.transform(y_test[..., i_dim:i_dim + 1])
        elif context_base[-4:] == 'tctx':
            self.feature_tctx_dim = x_train.shape[-1]
            scale_dim = self.feature_tctx_dim
            for i_dim in range(0, scale_dim):
                self.tctx_scaler = self._get_scalar(self.scaler_type, x_train[..., i_dim:i_dim + 1], y_train[..., i_dim:i_dim + 1])
                x_train[..., i_dim:i_dim + 1] = self.tctx_scaler.transform(x_train[..., i_dim:i_dim + 1])
                y_train[..., i_dim:i_dim + 1] = self.tctx_scaler.transform(y_train[..., i_dim:i_dim + 1])
                x_val[..., i_dim:i_dim + 1] = self.tctx_scaler.transform(x_val[..., i_dim:i_dim + 1])
                y_val[..., i_dim:i_dim + 1] = self.tctx_scaler.transform(y_val[..., i_dim:i_dim + 1])
                x_test[..., i_dim:i_dim + 1] = self.tctx_scaler.transform(x_test[..., i_dim:i_dim + 1])
                y_test[..., i_dim:i_dim + 1] = self.tctx_scaler.transform(y_test[..., i_dim:i_dim + 1])

        # 把训练集的X和y聚合在一起成为list，测试集验证集同理
        # x_train/y_train: (num_samples, input_length, ..., feature_dim)
        # train_data(list): train_data[i]是一个元组，由x_train[i]和y_train[i]组成
        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        return train_data, eval_data, test_data
