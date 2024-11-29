# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 20:08:46 2022

@author: yatzhang
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from libcity.utils import ensure_dir
from torch.utils.data import Dataset
from libcity.data.dataset import TrafficStateDataset
from libcity.data.utils import context_data_padding, generate_dataloader_context
from libcity.pipeline.embedkg_template import generate_spatial_kg, generate_temporal_kg, kg_embedding


class Traffic_Context_Dataset(Dataset):
    # def __init__(self, goal_data, auxi_data=None, spat_data=None, temp_data=None):
    def __init__(self, goal_data, auxi_data=None):
        self.goal_data = goal_data
        self.auxi_data = auxi_data
        # self.spat_data = spat_data
        # self.temp_data = temp_data

    def __len__(self):
        return len(self.goal_data)

    # Style in LibCity
    def __getitem__(self, idx):
        # goal
        goal = self.goal_data[idx][0]
        target_goal = self.goal_data[idx][1]
        # auxiliary
        auxi = self.auxi_data[idx][0]
        target_auxi = self.auxi_data[idx][1]
        # spatkg
        # spat = self.spat_data[idx][0]
        # target_spat = self.spat_data[idx][1]
        # tempkg
        # temp = self.temp_data[idx][0]
        # target_temp = self.temp_data[idx][1]
        # return goal, target_goal, auxi, target_auxi, spat, target_spat, temp, target_temp
        return goal, target_goal, auxi, target_auxi


class TrafficStateContextKGDataset(TrafficStateDataset):
    def __init__(self, config):
        super().__init__(config)

        self.context_feature_name = None
        self.feature_total_dim = 0
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = None, None, None
        self.parameters_str = str(self.dataset) + '_' + str(self.input_window) + '_' + str(self.output_window) + '_' \
                              + str(self.train_rate) + '_' + str(self.eval_rate) + '_' + str(self.scaler_type)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'point_based_{}.npz'.format(self.parameters_str))

    def _load_geo(self):
        """
        loading geo file, format [geo_id, type, coordinates, properties(several columns)]
        """
        super()._load_geo()

    def _load_rel(self):
        """
        loading rel file, format [rel_id, type, origin_id, destination_id, properties(several columns)]
        """
        super()._load_rel()

    def _load_dyna(self, filename):
        """
        loading dyna file, format[dyna_id, type, time, entity_id, properties(several columns)]
        """
        return super()._load_dyna_3d(filename)

    def datetime_to_float(self, _str_time):
        datetime_obj = datetime.strptime(_str_time.strip(), '%Y-%m-%dT%H:%M:%SZ')
        return time.mktime(datetime_obj.timetuple())

    def load_kg_auxi_dyna(self, filename, context_base):
        """
        loading dyna files，format [dyna_id, type, time, entity_id, properties(若干列)]

        Args:
            filename(str): filename without suffix
            context_base(str): used to generate the full filename based on filename(str)

        Returns:
            np.ndarray: 3d-array data (len_time, num_nodes, feature_dim), feature_dim includes datetime and entityid
        """
        # load datasets
        context_name = filename
        self._logger.info("Loading file " + context_name + '_' + context_base + '.dyna')
        dynafile = pd.read_csv(self.data_path + context_name + '.dyna', low_memory=False)
        dynafile = dynafile[dynafile.columns[2:]]  # all columns from 'time'
        dynafile = dynafile[dynafile.columns[:-2]]  # excluding 'traffic_speed' & 'count'
        # obtain time series
        self.timesolts = list(dynafile['time'][:int(dynafile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not dynafile['time'].isna().any():  # there should not be null value in time
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # convert to 3-d array
        feature_dim = len(dynafile.columns)  # include datetime and entityid
        df = dynafile[dynafile.columns[-feature_dim:]]
        # convert datetime and id to float
        df['time'] = df['time'].apply(self.datetime_to_float)
        df['entity_id'] = df['entity_id'].astype(int)
        len_time = len(self.timesolts)
        data = []
        for i in range(0, df.shape[0], len_time):
            origin_df = df[i:i + len_time].values
            if origin_df.shape[1] != 2:
                self._logger.info('[ERROR]The shape[1] of origin_df is {}, not 2'.format(origin_df.shape[1]))
                exit()
            # convert long_num into two parts
            conver_df = np.empty((origin_df.shape[0], origin_df.shape[1]+2), dtype=np.int)
            for j in range(origin_df.shape[0]):
                long_num = int(origin_df[j, 0])
                str_num = str(long_num)
                middle_index = len(str_num) // 2
                part1 = int(str_num[:middle_index])
                part2 = int(str_num[middle_index:])
                conver_df[j, 0] = part1
                conver_df[j, 1] = part2
                conver_df[j, 2] = len(str_num)
                conver_df[j, 3] = origin_df[j, 1]
            data.append(conver_df)
        # data = np.array(data, dtype=np.float)  # (len(self.geo_ids), len_time, feature_dim), only datetime & entity
        data = np.array(data, dtype=np.int)  # (len(self.geo_ids), len_time, feature_dim), only datetime & entity
        data = data.swapaxes(0, 1)  # (len_time, len(self.geo_ids), feature_dim)
        self._logger.info("Loaded file " + context_name + '_' + context_base + '.dyna' + ', shape=' + str(data.shape))
        return data

    def load_kg_data_dyna(self, filename, context_base):
        """
        loading dyna files，format [dyna_id, type, time, entity_id, properties(若干列)]

        Args:
            filename(str): filename without suffix
            context_base(str): used to generate the full filename based on filename(str)

        Returns:
            np.ndarray: 3d-array data (len_time, num_nodes, feature_dim), feature_dim includes datetime and entityid
        """
        # load datasets
        context_name = filename
        self._logger.info("Loading file " + context_name + '_' + context_base + '.dyna')
        dynafile = pd.read_csv(self.data_path + context_name + '.dyna', low_memory=False)
        dynafile = dynafile[dynafile.columns[2:]]  # all columns from 'time'
        dynafile = dynafile[dynafile.columns[:-1]]  # excluding 'count'
        # obtain time series
        self.timesolts = list(dynafile['time'][:int(dynafile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not dynafile['time'].isna().any():  # there should not be null value in time
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # convert to 3-d array
        feature_dim = len(dynafile.columns) - 2  # not include datetime and entityid
        df = dynafile[dynafile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i:i + len_time].values)
        data = np.array(data, dtype=np.float)  # (len(self.geo_ids), len_time, feature_dim), only datetime & entity
        data = data.swapaxes(0, 1)  # (len_time, len(self.geo_ids), feature_dim)
        self._logger.info("Loaded file " + context_name + '_' + context_base + '.dyna' + ', shape=' + str(data.shape))
        return data

    def get_data(self): 
        """
        return DataLoader, including training, validation, and test datasets

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        # process multi-files
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:  # str
            data_files = [self.data_files].copy()
        if len(data_files) > 1:
            self._logger.info("[ERROR] Multi-files encountered, but only processing one file")
            exit()
        filename = data_files[0]

        # load data
        df_goal = self.load_kg_data_dyna(filename, context_base='goal')  # (len_time, len(self.geo_ids), feature_dim)
        df_auxi = self.load_kg_auxi_dyna(filename, context_base='auxi')  # (len_time, len(self.geo_ids), feature_dim)

        # dataset organization
        x_train_goal, y_train_goal, x_val_goal, y_val_goal, x_test_goal, y_test_goal = None, None, None, None, None, None
        x_train_auxi, y_train_auxi, x_val_auxi, y_val_auxi, x_test_auxi, y_test_auxi = None, None, None, None, None, None
        # x_train_spat, y_train_spat, x_val_spat, y_val_spat, x_test_spat, y_test_spat = None, None, None, None, None, None
        # x_train_temp, y_train_temp, x_val_temp, y_val_temp, x_test_temp, y_test_temp = None, None, None, None, None, None
        if self.data is None:
            self.data = {}
            origin_cache_file_name = self.cache_file_name
            goal_cache_file_name = origin_cache_file_name[:-4] + '_goal.npz'  # goal
            auxi_cache_file_name = origin_cache_file_name[:-4] + '_auxi.npz'  # goal
            if self.cache_dataset and os.path.exists(goal_cache_file_name):
                x_train_goal, y_train_goal, x_val_goal, y_val_goal, x_test_goal, y_test_goal = self._load_cache_train_val_test_context(context_base='goal')
            else:
                x_train_goal, y_train_goal, x_val_goal, y_val_goal, x_test_goal, y_test_goal = self._generate_train_val_test_context(df_goal, context_base='goal')
            # auxi
            if self.cache_dataset and os.path.exists(auxi_cache_file_name):
                x_train_auxi, y_train_auxi, x_val_auxi, y_val_auxi, x_test_auxi, y_test_auxi = self._load_cache_train_val_test_context(context_base='auxi')
            else:
                x_train_auxi, y_train_auxi, x_val_auxi, y_val_auxi, x_test_auxi, y_test_auxi = self._generate_train_val_test_context(df_auxi, context_base='auxi')
            # spat and temp: donot save cached file as embeddings change a lot
            # x_train_spat, y_train_spat, x_val_spat, y_val_spat, x_test_spat, y_test_spat = self._generate_train_val_test_context(df_spat, context_base='spat')
            # x_train_temp, y_train_temp, x_val_temp, y_val_temp, x_test_temp, y_test_temp = self._generate_train_val_test_context(df_temp, context_base='temp')

        # Data scaling for goal, auxi data doesnot need to scale (string)
        goal_train_data, goal_val_data, goal_test_data = self._scaler_data(x_train_goal, y_train_goal, x_val_goal, y_val_goal, x_test_goal, y_test_goal, 'goal')
        auxi_train_data, auxi_val_data, auxi_test_data = self._nonscaler_data(x_train_auxi, y_train_auxi, x_val_auxi, y_val_auxi, x_test_auxi, y_test_auxi, 'auxi')
        # spat_train_data, spat_val_data, spat_test_data = self._nonscaler_data(x_train_spat, y_train_spat, x_val_spat, y_val_spat, x_test_spat, y_test_spat, 'spat')
        # temp_train_data, temp_val_data, temp_test_data = self._nonscaler_data(x_train_temp, y_train_temp, x_val_temp, y_val_temp, x_test_temp, y_test_temp, 'auxi')

        # when last batch is less than batch_size, whether we need to make it up using the last element
        if self.pad_with_last_sample:
            goal_train_data, goal_val_data, goal_test_data = context_data_padding(goal_train_data, goal_val_data,
                                                                                  goal_test_data, self.batch_size)
            auxi_train_data, auxi_val_data, auxi_test_data = context_data_padding(auxi_train_data, auxi_val_data,
                                                                                  auxi_test_data, self.batch_size)
            # spat_train_data, spat_val_data, spat_test_data = context_data_padding(spat_train_data, spat_val_data,
            #                                                                       spat_test_data, self.batch_size)
            # temp_train_data, temp_val_data, temp_test_data = context_data_padding(temp_train_data, temp_val_data,
            #                                                                       temp_test_data, self.batch_size)

        # context_feature_name
        self.context_feature_name = {'X_goal': 'float', 'y_goal': 'float',
                                     'X_auxi': 'float', 'y_auxi': 'float'}
                                     # 'X_spat': 'float', 'y_spat': 'float',
                                     # 'X_temp': 'float', 'y_temp': 'float'}

        # build training, validation, and test dataset
        train_dataset = Traffic_Context_Dataset(goal_train_data, auxi_train_data)  # ,spat_train_data, temp_train_data)
        val_dataset = Traffic_Context_Dataset(goal_val_data, auxi_val_data)  # ,spat_val_data, temp_val_data)
        test_dataset = Traffic_Context_Dataset(goal_test_data, auxi_test_data)  # ,spat_test_data, temp_test_data)

        # create dataloader for loading datasets into models
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader_context(train_dataset, val_dataset, test_dataset,
                                        self.context_feature_name, self.batch_size, self.num_workers,
                                        shuffle=False, pad_with_last_sample=self.pad_with_last_sample)

        # The format of train_dataloader: same as Traffic_Context_Dataset, i.e., 2067-4-12-156-1, but two differences
        # Dif 1: the 1st dim, i.e., the total length of 2067 is replaced by the batch_size, e.g., 8
        # Dif 2: the 2nd dim, i.e., 4 is replaced by the key of 'X_goal', 'y_goal', 'X_auxi', 'y_auxi'
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
        return feature of datatsets
        scaler is normalization method, adj_mx is adjacency matrix, num_nodes is num of nodes
        feature_dim is the dimension of input features, output_dim is the dimension of output result

        Returns:
            dict
        """
        self.feature_total_dim = self.feature_dim
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "ext_dim": self.ext_dim,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_total_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches}

    def get_kge_template(self):
        # process multi-files
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:  # str
            data_files = [self.data_files].copy()
        if len(data_files) > 1:
            self._logger.info("[ERROR] Multi-files encountered, but only processing one file")
            exit()
        filename = data_files[0]

        # load data
        df_goal = self.load_kg_data_dyna(filename, context_base='goal')  # (len_time, len(self.geo_ids), feature_dim)
        df_auxi = self.load_kg_auxi_dyna(filename, context_base='auxi')  # (len_time, len(self.geo_ids), feature_dim)

        # road list
        # roadlab_list = []
        # for _elem_id in list(df_auxi[0, :, 3]):  # road_id list
        #     roadlab_list.append('road_' + str(int(_elem_id)))

        ##############################################################################################################
        # obtain spatial embeddings
        spat_kg_model = self.config.get('spat_model_used')
        df_spat_shape = list(df_goal.shape)
        df_spat_shape[-1] = self.config.get('kg_embed_dim')
        # generate spat tf
        tf_spat = generate_spatial_kg(self.config, self._logger)
        # road_spat_id = kg_entity_id_from_list(tf_spat, roadlab_list)
        # spat_kge path
        # spat_file = 'spat_kge_' + spat_kg_model + '_spat[{}]'.format(self.config.get('spat_cont_used')) \
        #             + '_buffer[{}]'.format(self.config.get('spat_buffer_used')) + '_link[{}]'.format(self.config.get('spat_link_used')) \
        #             + '_red[{}-{}-{}]_template'.format(self.config.get('seed'), self.config.get('kg_embed_dim'), self.config.get('kg_epochs_num'))
        spat_file = 'spat_kge_' + spat_kg_model + '_spat[{}]'.format(self.config.get('spat_cont_used')) \
                    + '_buffer[{}]'.format(self.config.get('spat_buffer_used')) + '_link[{}]'.format(self.config.get('spat_link_used')) \
                    + '_red[{}-{}-{}]_template'.format(1, self.config.get('kg_embed_dim'), self.config.get('kg_epochs_num'))
        spat_kge_path = os.path.join('./raw_data/{}'.format(self.config.get('dataset')), '{}'.format(spat_file))
        spat_kge_ent_pickle = os.path.join('./raw_data/{}'.format(self.config.get('dataset')), '{}/spat_kge_ent.pickle'.format(spat_file))
        spat_kge_rel_pickle = os.path.join('./raw_data/{}'.format(self.config.get('dataset')), '{}/spat_kge_rel.pickle'.format(spat_file))
        # obtain spat_kge
        if os.path.exists(spat_kge_ent_pickle):
            with open(spat_kge_ent_pickle, 'rb') as f_pickle1:
                spat_kge_ent = pickle.load(f_pickle1)
            with open(spat_kge_rel_pickle, 'rb') as f_pickle2:
                spat_kge_rel = pickle.load(f_pickle2)
        else:
            if not os.path.exists(spat_kge_path):
                os.makedirs(spat_kge_path)
            spat_kge_ent, spat_kge_rel = kg_embedding(tf_spat, self.config, self._logger, spat_kg_model)
            with open(spat_kge_ent_pickle, 'wb') as f_pickle1:
                pickle.dump(spat_kge_ent, f_pickle1, protocol=4)
            with open(spat_kge_rel_pickle, 'wb') as f_pickle2:
                pickle.dump(spat_kge_rel, f_pickle2, protocol=4)
        # generate df_spat embeddings
        # road_spat_embed = spat_kge_ent[road_spat_id, :]
        # df_spat = np.zeros(df_spat_shape, dtype=np.float)
        # for i0 in range(df_spat.shape[0]):  # each time step
        #     df_spat[i0] = road_spat_embed
        self._logger.info("***[SPAT] kg ent spat_embed with max:{} and min{}...".format(np.max(spat_kge_ent), np.min(spat_kge_ent)))

        ##############################################################################################################
        # obtain temporal embeddings
        temp_kg_model = self.config.get('temp_model_used')
        df_temp_shape = list(df_goal.shape)
        df_temp_shape[-1] = self.config.get('kg_embed_dim')
        # generate temp tf
        tf_temp = generate_temporal_kg(self.config, self._logger)
        # road_temp_id = kg_entity_id_from_list(tf_temp, roadlab_list)
        # temp_kge path
        # temp_file = 'temp_kge_' + temp_kg_model + '_temp[{}]'.format(self.config.get('temp_cont_used')) + \
        #             '_time[{}]'.format(self.config.get('temp_time_used')) + '_link[{}]'.format(self.config.get('temp_link_used')) \
        #             + '_red[{}-{}-{}]_template'.format(self.config.get('seed'), self.config.get('kg_embed_dim'), self.config.get('kg_epochs_num'))
        temp_file = 'temp_kge_' + temp_kg_model + '_temp[{}]'.format(self.config.get('temp_cont_used')) + \
                    '_time[{}]'.format(self.config.get('temp_time_used')) + '_link[{}]'.format(self.config.get('temp_link_used')) \
                    + '_red[{}-{}-{}]_template'.format(1, self.config.get('kg_embed_dim'), self.config.get('kg_epochs_num'))
        temp_kge_path = os.path.join('./raw_data/{}'.format(self.config.get('dataset')), '{}'.format(temp_file))
        temp_kge_ent_pickle = os.path.join('./raw_data/{}'.format(self.config.get('dataset')), '{}/temp_kge_ent.pickle'.format(temp_file))
        temp_kge_rel_pickle = os.path.join('./raw_data/{}'.format(self.config.get('dataset')), '{}/temp_kge_rel.pickle'.format(temp_file))
        # obtain temp_kge
        if os.path.exists(temp_kge_ent_pickle):
            with open(temp_kge_ent_pickle, 'rb') as f_pickle3:
                temp_kge_ent = pickle.load(f_pickle3)
            with open(temp_kge_rel_pickle, 'rb') as f_pickle4:
                temp_kge_rel = pickle.load(f_pickle4)
        else:
            if not os.path.exists(temp_kge_path):
                os.makedirs(temp_kge_path)
            temp_kge_ent, temp_kge_rel = kg_embedding(tf_temp, self.config, self._logger, temp_kg_model)
            with open(temp_kge_ent_pickle, 'wb') as f_pickle3:
                pickle.dump(temp_kge_ent, f_pickle3, protocol=4)
            with open(temp_kge_rel_pickle, 'wb') as f_pickle4:
                pickle.dump(temp_kge_rel, f_pickle4, protocol=4)
        # generate df_temp embeddings
        # road_temp_embed = temp_kge_ent[road_temp_id, :]
        # df_temp = np.zeros(df_temp_shape, dtype=np.float)
        # for i0 in range(df_temp.shape[0]):  # each time step
        #     df_temp[i0] = road_temp_embed
        self._logger.info("***[TEMP] kg ent temp_embed with max:{} and min{}...".format(np.max(temp_kge_ent), np.min(temp_kge_ent)))

        # return spat_kge and temp_kge
        return df_goal, df_auxi, spat_file, temp_file, spat_kge_ent, spat_kge_rel, temp_kge_ent, temp_kge_rel

    def _generate_train_val_test_context(self, df, context_base):
        """
        generate datsets, and split train, validation, and test datasets

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        x, y = self._generate_data_context(df, context_base)
        return self._split_train_val_test_context(x, y, context_base)
    
    def _generate_data_context(self, df, context_base):
        """
        load (.dyna/.grid/.od/.gridod)

        Returns:
            tuple: tuple contains:
                x(np.ndarray): input data，(num_samples, input_length, ..., feature_dim) \n
                y(np.ndarray): output data，(num_samples, output_length, ..., feature_dim)
        """
        x_list, y_list = [], []
        x, y = self._generate_input_data(df)
        # x: (num_samples, input_length, ..., input_dim)
        # y: (num_samples, output_length, ..., output_dim)
        x_list.append(x)
        y_list.append(y)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        self._logger.info("Dataset created of {}".format(context_base))
        self._logger.info("x shape: " + str(x.shape) + ", y shape: " + str(y.shape))
        return x, y
    
    def _split_train_val_test_context(self, x, y, context_base):
        """
        split train, validation, and test datsets, then generate cached datasets

        Args:
            x(np.ndarray): input data (num_samples, input_length, ..., feature_dim)
            y(np.ndarray): output data (num_samples, input_length, ..., feature_dim)

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

        if self.cache_dataset and context_base != 'spat' and context_base != 'temp':
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
        load cached train, validation, and test datasets

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
        return DataLoader

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        # scale_dim = self.output_dim
        if context_base == 'goal':
            self.feature_dim = x_train.shape[-1]
            scale_dim = self.output_dim
            self.scaler = self._get_scalar(self.scaler_type, x_train[..., :scale_dim], y_train[..., :scale_dim])
            x_train[..., :scale_dim] = self.scaler.transform(x_train[..., :scale_dim])
            y_train[..., :scale_dim] = self.scaler.transform(y_train[..., :scale_dim])
            x_val[..., :scale_dim] = self.scaler.transform(x_val[..., :scale_dim])
            y_val[..., :scale_dim] = self.scaler.transform(y_val[..., :scale_dim])
            x_test[..., :scale_dim] = self.scaler.transform(x_test[..., :scale_dim])
            y_test[..., :scale_dim] = self.scaler.transform(y_test[..., :scale_dim])
        
        # combine x&y in training dataset into a list, same for validation and test datasets
        # x_train/y_train: (num_samples, input_length, ..., feature_dim)
        # train_data(list): train_data[i] is a tuple, consisting of x_train[i] and y_train[i]
        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        self._logger.info("Scaler of {} completed".format(context_base))
        return train_data, eval_data, test_data

    def _scaler_data_by_column(self, x_train, y_train, x_val, y_val, x_test, y_test, context_base):
        """
        return DataLoader

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        # scale_dim = self.output_dim
        if context_base == 'goal':
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

        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        self._logger.info("Scaler_column of {} completed".format(context_base))
        return train_data, eval_data, test_data

    def _nonscaler_data(self, x_train, y_train, x_val, y_val, x_test, y_test, context_base):
        """
        return DataLoader

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        # combine x&y in training dataset into a list, same for validation and test datasets
        # x_train/y_train: (num_samples, input_length, ..., feature_dim)
        # train_data(list): train_data[i] is a tuple, consisting of x_train[i] and y_train[i]
        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        self._logger.info("Non scaler of {} completed".format(context_base))
        return train_data, eval_data, test_data