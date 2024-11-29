import copy
import os
import platform
import numpy as np
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import FIFOScheduler, ASHAScheduler, MedianStoppingRule
from ray.tune.suggest import ConcurrencyLimiter
import json
import torch
import random
import pickle
from datetime import datetime, time
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model, get_logger, ensure_dir, set_random_seed
from libcity.pipeline.embedkg_template import obatin_spatial_pickle, obatin_temporal_pickle
from libcity.pipeline.embedkg_template import generate_spatial_kg, generate_temporal_kg
from libcity.pipeline.embedkg_template import generate_kgsub_spat, generate_kgsub_temp_notcover


def cal_ent_via_rel(config, logger, model, ent, rel, weight):
    ent_change = []
    if config['kg_weight'] == 'add':
        if model == 'ComplEx':
            ent_change = np.dot(np.diag(rel), ent) + weight
            ent_change = np.real(ent_change)
        elif model == 'KG2E':
            ent_change = ent - rel + weight
        elif model == 'AutoSF':
            ent_change = np.dot(np.diag(rel), ent) + weight
        else:
            logger.info('[ERROR]model-{} was not used in embeddings'.format(model))
    elif config['kg_weight'] == 'times':
        if model == 'ComplEx':
            ent_change = weight * np.dot(np.diag(rel), ent)
            ent_change = np.real(ent_change)
        elif model == 'KG2E':
            ent_change = weight * (ent - rel)
        elif model == 'AutoSF':
            ent_change = weight * np.dot(np.diag(rel), ent)
        else:
            logger.info('[ERROR]model-{} was not used in embeddings'.format(model))
    return np.array(ent_change)


def obtain_temp_kge_final(config, logger, x_goal, x_auxi, dict_kge, temp_datetime):
    # basic par
    temp_attr_used = config.get('temp_attr_used')
    temp_model_used = config.get('temp_model_used')
    embed_dim = config.get('kg_embed_dim')
    kg_context = config.get('kg_context')
    temp_ent_kge, temp_rel_kge = dict_kge['temp_ent_kge'], dict_kge['temp_rel_kge']
    temp_ent_label, temp_rel_label = dict_kge['temp_ent_label'], dict_kge['temp_rel_label']

    # sub_kg for temp
    subdict_temp = dict_kge['sub_temp'][temp_datetime]  # {road_id: {road/poi/land/link: [fact]}}
    temp_attr_used_list = temp_attr_used.split('-')

    # dict_final_temp
    subdict_temp_kge = {}
    # generate temp embeddings
    random_integer = random.randint(1, 1000)
    for _dim2 in range(x_goal.shape[1]):  # num_nodes
        # road_str = 'road_' + str(int(x_auxi[_dim1, _dim2, _dim3, 3]))
        road_str = 'road_' + str(int(x_auxi[0, _dim2, 3]))
        # temp embedding, 3*embed: 1-itself, 2-cont, 3-link
        temp_embed_1 = np.zeros(embed_dim, dtype=np.float)
        temp_embed_2 = np.zeros(embed_dim * 3, dtype=np.float)  # time/jam/weather
        embed2_time = np.zeros(embed_dim, dtype=np.float)
        embed2_jam = np.zeros(embed_dim, dtype=np.float)
        embed2_weather = np.zeros(embed_dim, dtype=np.float)
        temp_embed_3 = np.zeros(embed_dim * 3, dtype=np.float)
        embed3_hour = np.zeros(embed_dim, dtype=np.float)
        embed3_day = np.zeros(embed_dim, dtype=np.float)
        embed3_week = np.zeros(embed_dim, dtype=np.float)
        _subkg_fact = []
        if 'time' in temp_attr_used_list:
            _subkg_fact.extend(subdict_temp[road_str]['time'])
        if 'jam' in temp_attr_used_list:
            _subkg_fact.extend(subdict_temp[road_str]['jam'])
        if 'weather' in temp_attr_used_list:
            _subkg_fact.extend(subdict_temp[road_str]['tprt'])
            _subkg_fact.extend(subdict_temp[road_str]['rain'])
            _subkg_fact.extend(subdict_temp[road_str]['wind'])
        _subkg_fact_temp = []
        _subkg_fact_link = []
        for _fact in _subkg_fact:  # temp link
            if 'Hourly' in _fact[1] or 'Daily' in _fact[1] or 'Weekly' in _fact[1]:
                _subkg_fact_link.append(_fact)
            else:  # temp non_link
                _subkg_fact_temp.append(_fact)
        # embed 1
        if temp_model_used == 'ComplEx':
            temp_embed_1 = np.real(np.array(temp_ent_kge[temp_ent_label[road_str], :]))
        else:
            temp_embed_1 = np.array(temp_ent_kge[temp_ent_label[road_str], :])
        # embed 2
        for _fact in _subkg_fact_temp:
            _weight = float(_fact[3])
            if 'hourOfDay' in _fact[1] or 'dayOfWeek' in _fact[1] or 'isRestWork' in _fact[1]:
                embed2_time += cal_ent_via_rel(config, logger, temp_model_used, temp_ent_kge[temp_ent_label[_fact[2]], :],
                                              temp_rel_kge[temp_rel_label[_fact[1]], :], _weight)
            if 'hasjamCurrent' in _fact[1] or 'hasjamAve' in _fact[1]:
                embed2_jam += cal_ent_via_rel(config, logger, temp_model_used, temp_ent_kge[temp_ent_label[_fact[2]], :],
                                             temp_rel_kge[temp_rel_label[_fact[1]], :], _weight)
            if 'hastprtAve' in _fact[1] or 'hasrainAve' in _fact[1] or 'haswindAve' in _fact[1] or \
                    'hastprtCurrent' in _fact[1] or 'hasrainCurrent' in _fact[1] or 'haswindCurrent' in _fact[1]:
                embed2_weather += cal_ent_via_rel(config, logger, temp_model_used, temp_ent_kge[temp_ent_label[_fact[2]], :],
                                                 temp_rel_kge[temp_rel_label[_fact[1]], :], _weight)
        temp_embed_2[: embed_dim] = embed2_time
        temp_embed_2[embed_dim: 2 * embed_dim] = embed2_jam
        temp_embed_2[2 * embed_dim: 3 * embed_dim] = embed2_weather
        # embed 3
        for _fact in _subkg_fact_link:
            _weight = float(_fact[3])
            if 'Hourly' in _fact[1]:
                embed3_hour += cal_ent_via_rel(config, logger, temp_model_used, temp_ent_kge[temp_ent_label[_fact[2]], :],
                                             temp_rel_kge[temp_rel_label[_fact[1]], :], _weight)
            if 'Daily' in _fact[1]:
                embed3_day += cal_ent_via_rel(config, logger, temp_model_used, temp_ent_kge[temp_ent_label[_fact[2]], :],
                                             temp_rel_kge[temp_rel_label[_fact[1]], :], _weight)
            if 'Weekly' in _fact[1]:
                embed3_week += cal_ent_via_rel(config, logger, temp_model_used, temp_ent_kge[temp_ent_label[_fact[2]], :],
                                             temp_rel_kge[temp_rel_label[_fact[1]], :], _weight)
        temp_embed_3[: embed_dim] = embed3_hour
        temp_embed_3[embed_dim: 2 * embed_dim] = embed3_day
        temp_embed_3[2 * embed_dim: 3 * embed_dim] = embed3_week
        final_embed = np.concatenate((temp_embed_1, temp_embed_2, temp_embed_3), axis=0)
        subdict_temp_kge[road_str] = final_embed
        if _dim2 == random_integer and temp_datetime.time() == time(0, 5, 0):
            logger.info('{}, {} temp_embed1:{}'.format(temp_datetime, road_str, temp_embed_1))
            logger.info('{}, {} temp_embed2:{}'.format(temp_datetime, road_str, temp_embed_2))
            logger.info('{}, {} temp_embed3:{}'.format(temp_datetime, road_str, temp_embed_3))
    return subdict_temp_kge


def convert2datetime(x_auxi_slice):
    part1, part2, total_length = map(int, x_auxi_slice)
    long_num = part1 * 10 ** (total_length - len(str(part1))) + part2
    return datetime.fromtimestamp(long_num)


def process_temp_datetime_notcover(config, logger, temp_datetime, temp_datetime_list, np_goal, np_auxi, dictKG_temporal, dict_kge, kg_weight_temp):
    subdict_temp = generate_kgsub_temp_notcover(config, logger, dictKG_temporal, temp_datetime, temp_datetime_list, kg_weight_temp)
    dict_kge['sub_temp'][temp_datetime] = copy.deepcopy(subdict_temp)
    subdict_temp_kge = obtain_temp_kge_final(config, logger, np_goal, np_auxi, dict_kge, temp_datetime)
    dict_kge['sub_temp_emd'][temp_datetime] = copy.deepcopy(subdict_temp_kge)
    if temp_datetime.time() == time(0, 5, 0):
        logger.info('[During MP Complete]:{}'.format(temp_datetime))


def split_list(lst, sub):
    quotient, remainder = divmod(len(lst), sub)
    start = 0
    result = []
    for i in range(sub):
        end = start + quotient + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end
    return result


def run_model(task=None, model_name=None, dataset_name=None, config_file=None,
              saved_model=True, train=True, other_args=None):
    """
    Args:
        task(str): task name
        model_name(str): model name
        dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        saved_model(bool): whether to save the model
        train(bool): whether to train the model
        other_args(dict): the rest parameter args, which will be pass to the Config
    """
    # load config
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args)
    exp_id = config.get('exp_id', None)
    if exp_id is None:
        # Make a new experiment ID
        exp_id = int(random.SystemRandom().random() * 100000)
        config['exp_id'] = exp_id
    # logger
    logger = get_logger(config)
    logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(exp_id)))
    config['weight_decay'] = config['wd_set']
    config['cl_decay_steps'] = config['set_cl_decay_steps']
    config['max_diffusion_step'] = config['set_max_diffusion_step']
    config['num_rnn_layers'] = config['set_num_rnn_layers']
    config['rnn_units'] = config['set_rnn_units']
    config['bidir_adj_mx'] = config['set_bidir_adj_mx']
    logger.info(config.config)
    # seed
    seed = config.get('seed', 0)
    set_random_seed(seed)
    # load dataset
    dataset = get_dataset(config)
    # split dataset
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    # load model and executor
    model_cache_file = './libcity/cache/{}/model_cache/{}_{}.m'.format(
        exp_id, model_name, dataset_name)
    model = get_model(config, data_feature)
    executor = get_executor(config, model, data_feature)
    # whether use kg
    kg_switch = config.get('kg_switch')
    kg_context = config.get('kg_context')
    if kg_switch:
        # obtain dictKG_spatial, dictKG_temporal
        dictKG_spatial = obatin_spatial_pickle(config, logger)
        dictKG_temporal = obatin_temporal_pickle(config, logger)
        # obtan entity and rel id
        tf_spat = generate_spatial_kg(config, logger)
        spat_ent_label = tf_spat.entity_labeling.label_to_id
        spat_rel_label = tf_spat.relation_labeling.label_to_id
        tf_temp = generate_temporal_kg(config, logger)
        temp_ent_label = tf_temp.entity_labeling.label_to_id
        temp_rel_label = tf_temp.relation_labeling.label_to_id
        # obtain subgraphs
        subdict_spat = generate_kgsub_spat(config, logger, dictKG_spatial)
        subdict_temp = {}
        # np_goal/np_auxi: (len_time, len(self.geo_ids), feature_dim)
        np_goal, np_auxi, spat_file, temp_file, spat_ent_kge, spat_rel_kge, temp_ent_kge, temp_rel_kge = dataset.get_kge_template()
        subdict_spat_kge, subdict_temp_kge = {}, {}
        # temp_pickle_file
        temp_pickle_file = config.get('temp_pickle_file')
        with open(temp_pickle_file, 'rb') as f_pickle:
            dictKG_temporal = pickle.load(f_pickle)  # dict_dynamic {time: {id: keys: {value}}}
        temp_datetime_list = []
        for _dim1 in range(np_goal.shape[0]):  # len_time
            temp_datetime = convert2datetime(np_auxi[_dim1, 0, 0:3])  # convert two parts into long_num
            temp_datetime_list.append(temp_datetime)
        temp_datetime_list = list(dict.fromkeys(temp_datetime_list))
        logger.info('temp_datetime from[{}] end[{}]'.format(temp_datetime_list[0], temp_datetime_list[-1]))

        # organize all data
        dict_kge = {'dict_spat': dictKG_spatial, 'dict_temp': dictKG_temporal,  # origin dict
                    'spat_ent_kge': spat_ent_kge, 'spat_rel_kge': spat_rel_kge,  # spat kg embedding
                    'temp_ent_kge': temp_ent_kge, 'temp_rel_kge': temp_rel_kge,  # temp kg embedding
                    'spat_ent_label': spat_ent_label, 'spat_rel_label': spat_rel_label,  # spat label
                    'temp_ent_label': temp_ent_label, 'temp_rel_label': temp_rel_label,  # temp label
                    'sub_spat': subdict_spat, 'sub_temp': subdict_temp,  # sub kg for spat and temp
                    'sub_spat_emd': subdict_spat_kge, 'sub_temp_emd': subdict_temp_kge}  # sub embed for spat and temp
        # integration flag
        if config['kg_weight'] == 'add':
            kg_weight_temp = 'add'
            temp_kge_emd_file = kg_weight_temp + '_attr[{}]'.format(config.get('temp_attr_used')) + \
                                '_time[{}]'.format(config.get('temp_time_attr')) + \
                                '_link[{}]'.format(config.get('temp_link_attr'))
        else:
            kg_weight_temp = 'times'
            temp_kge_emd_file = 'attr[{}]'.format(config.get('temp_attr_used')) + \
                                '_time[{}]'.format(config.get('temp_time_attr')) + \
                                '_link[{}]'.format(config.get('temp_link_attr'))
        # pickle file name
        temp_kge_emd_pickle = os.path.join('./raw_data/{}'.format(config.get('dataset')), '{}/type_temp_kge_emd_notcover_{}.pickle'.format(temp_file, temp_kge_emd_file))
        if platform.system() == "Windows":
            temp_kge_emd_pickle = r'\\?\\' + os.path.abspath(temp_kge_emd_pickle)
        if kg_context != 'spat':
            if os.path.exists(temp_kge_emd_pickle):
                with open(temp_kge_emd_pickle, 'rb') as f_pickle:
                    dict_kge_part = pickle.load(f_pickle)
                    dict_kge['sub_temp'] = copy.deepcopy(dict_kge_part['sub_temp'])
                    dict_kge['sub_temp_emd'] = copy.deepcopy(dict_kge_part['sub_temp_emd'])
                logger.info('[MP]Load successfully from pickle')
            else:
                temp_datetime_list = []
                for _dim1 in range(np_goal.shape[0]):  # len_time
                    temp_datetime = convert2datetime(np_auxi[_dim1, 0, 0:3])  # convert two parts into long_num
                    temp_datetime_list.append(temp_datetime)
                temp_datetime_list = list(dict.fromkeys(temp_datetime_list))
                logger.info('temp_datetime from[{}] end[{}]'.format(temp_datetime_list[0], temp_datetime_list[-1]))
                for temp_datetime in temp_datetime_list:
                    process_temp_datetime_notcover(config, logger, temp_datetime, temp_datetime_list, np_goal, np_auxi, dictKG_temporal, dict_kge, kg_weight_temp)
                    if temp_datetime.time() == time(0, 5, 0):
                        logger.info('[Test MP Complete]:{}'.format(temp_datetime))
                        logger.info('[Test MP Complete]:dict_kge[sub_temp][temp_datetime]-{},{}'.format(type(dict_kge['sub_temp'][temp_datetime]), len(dict_kge['sub_temp'][temp_datetime])))
                        logger.info('[Test MP Complete]:dict_kge[sub_temp_emd][temp_datetime]-{},{}'.format(type(dict_kge['sub_temp_emd'][temp_datetime]), len(dict_kge['sub_temp_emd'][temp_datetime])))
                dict_kge_part = {}
                dict_kge_part['sub_temp'] = copy.deepcopy(dict_kge['sub_temp'])
                dict_kge_part['sub_temp_emd'] = copy.deepcopy(dict_kge['sub_temp_emd'])
                with open(temp_kge_emd_pickle, 'wb') as f_pickle:
                    pickle.dump(dict_kge_part, f_pickle, protocol=4)
                logger.info('[MP]Store successfully into pickle')
        # training
        if train or not os.path.exists(model_cache_file):
            executor.kg_train(train_data, valid_data, dict_kge)
            if saved_model:
                executor.save_model(model_cache_file)
        else:
            executor.load_model(model_cache_file)
        # evaluation: the result will be in the dir of cache/evaluate_cache
        executor.kg_evaluate(test_data, dict_kge)
    else:
        # training
        if train or not os.path.exists(model_cache_file):
            executor.train(train_data, valid_data)
            if saved_model:
                executor.save_model(model_cache_file)
        else:
            executor.load_model(model_cache_file)
        # evaluation: the result will be in the dir of cache/evaluate_cache
        executor.evaluate(test_data)


def parse_search_space(space_file):
    search_space = {}
    if os.path.exists('./{}.json'.format(space_file)):
        with open('./{}.json'.format(space_file), 'r') as f:
            paras_dict = json.load(f)
            for name in paras_dict:
                paras_type = paras_dict[name]['type']
                if paras_type == 'uniform':
                    # name type low up
                    try:
                        search_space[name] = tune.uniform(paras_dict[name]['lower'], paras_dict[name]['upper'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing uniform type.')
                elif paras_type == 'randn':
                    # name type mean sd
                    try:
                        search_space[name] = tune.randn(paras_dict[name]['mean'], paras_dict[name]['sd'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing randn type.')
                elif paras_type == 'randint':
                    # name type lower upper
                    try:
                        if 'lower' not in paras_dict[name]:
                            search_space[name] = tune.randint(paras_dict[name]['upper'])
                        else:
                            search_space[name] = tune.randint(paras_dict[name]['lower'], paras_dict[name]['upper'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing randint type.')
                elif paras_type == 'choice':
                    # name type list
                    try:
                        search_space[name] = tune.choice(paras_dict[name]['list'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing choice type.')
                elif paras_type == 'grid_search':
                    # name type list
                    try:
                        search_space[name] = tune.grid_search(paras_dict[name]['list'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing grid_search type.')
                else:
                    raise TypeError('The space file does not meet the format requirements,\
                            when parsing an undefined type.')
    else:
        raise FileNotFoundError('The space file {}.json is not found. Please ensure \
            the config file is in the root dir and is a txt.'.format(space_file))
    return search_space


def hyper_parameter(task=None, model_name=None, dataset_name=None, config_file=None, space_file=None,
                    scheduler=None, search_alg=None, other_args=None, num_samples=5, max_concurrent=1,
                    cpu_per_trial=1, gpu_per_trial=1):
    """ Use Ray tune to hyper parameter tune

    Args:
        task(str): task name
        model_name(str): model name
        dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        space_file(str): the file which specifies the parameter search space
        scheduler(str): the trial sheduler which will be used in ray.tune.run
        search_alg(str): the search algorithm
        other_args(dict): the rest parameter args, which will be pass to the Config
    """
    # load config
    experiment_config = ConfigParser(task, model_name, dataset_name, config_file=config_file,
                                     other_args=other_args)
    # exp_id
    exp_id = experiment_config.get('exp_id', None)
    if exp_id is None:
        exp_id = int(random.SystemRandom().random() * 100000)
        experiment_config['exp_id'] = exp_id
    # logger
    logger = get_logger(experiment_config)
    logger.info('Begin ray-tune, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(exp_id)))
    logger.info(experiment_config.config)
    # check space_file
    if space_file is None:
        logger.error('the space_file should not be None when hyperparameter tune.')
        exit(0)
    # seed
    seed = experiment_config.get('seed', 0)
    set_random_seed(seed)
    # parse space_file
    search_sapce = parse_search_space(space_file)
    # load dataset
    dataset = get_dataset(experiment_config)
    # get train valid test data
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()

    def train(config, checkpoint_dir=None, experiment_config=None,
              train_data=None, valid_data=None, data_feature=None):
        """trainable function which meets ray tune API

        Args:
            config (dict): A dict of hyperparameter.
        """
        # modify experiment_config
        for key in config:
            if key in experiment_config:
                experiment_config[key] = config[key]
        experiment_config['hyper_tune'] = True
        logger = get_logger(experiment_config)
        # exp_id
        exp_id = int(random.SystemRandom().random() * 100000)
        experiment_config['exp_id'] = exp_id
        logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                    format(str(task), str(model_name), str(dataset_name), str(exp_id)))
        logger.info('running parameters: ' + str(config))
        # load model
        model = get_model(experiment_config, data_feature)
        # load executor
        executor = get_executor(experiment_config, model, data_feature)
        # checkpoint by ray tune
        if checkpoint_dir:
            checkpoint = os.path.join(checkpoint_dir, 'checkpoint')
            executor.load_model(checkpoint)
        # train
        executor.train(train_data, valid_data)

    # init search algorithm and scheduler
    if search_alg == 'BasicSearch':
        algorithm = BasicVariantGenerator()
    elif search_alg == 'BayesOptSearch':
        algorithm = BayesOptSearch(metric='loss', mode='min')
        # add concurrency limit
        algorithm = ConcurrencyLimiter(algorithm, max_concurrent=max_concurrent)
    elif search_alg == 'HyperOpt':
        algorithm = HyperOptSearch(metric='loss', mode='min')
        # add concurrency limit
        algorithm = ConcurrencyLimiter(algorithm, max_concurrent=max_concurrent)
    else:
        raise ValueError('the search_alg is illegal.')
    if scheduler == 'FIFO':
        tune_scheduler = FIFOScheduler()
    elif scheduler == 'ASHA':
        tune_scheduler = ASHAScheduler()
    elif scheduler == 'MedianStoppingRule':
        tune_scheduler = MedianStoppingRule()
    else:
        raise ValueError('the scheduler is illegal')
    # ray tune run
    ensure_dir('./libcity/cache/hyper_tune')
    result = tune.run(tune.with_parameters(train, experiment_config=experiment_config, train_data=train_data,
                      valid_data=valid_data, data_feature=data_feature),
                      resources_per_trial={'cpu': cpu_per_trial, 'gpu': gpu_per_trial}, config=search_sapce,
                      metric='loss', mode='min', scheduler=tune_scheduler, search_alg=algorithm,
                      local_dir='./libcity/cache/hyper_tune', num_samples=num_samples)
    best_trial = result.get_best_trial("loss", "min", "last")
    logger.info("Best trial config: {}".format(best_trial.config))
    logger.info("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    # save best
    best_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
    model_state, optimizer_state = torch.load(best_path)
    model_cache_file = './libcity/cache/{}/model_cache/{}_{}.m'.format(
        exp_id, model_name, dataset_name)
    ensure_dir('./libcity/cache/{}/model_cache'.format(exp_id))
    torch.save((model_state, optimizer_state), model_cache_file)


def objective_function(task=None, model_name=None, dataset_name=None, config_file=None,
                       saved_model=True, train=True, other_args=None, hyper_config_dict=None):
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args, hyper_config_dict)
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()

    model = get_model(config, data_feature)
    executor = get_executor(config, model, data_feature)
    best_valid_score = executor.train(train_data, valid_data)
    test_result = executor.evaluate(test_data)

    return {
        'best_valid_score': best_valid_score,
        'test_result': test_result
    }
