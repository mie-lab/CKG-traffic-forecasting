"""
model for running and training
"""

import argparse

from libcity.pipeline import run_model
from libcity.utils import str2bool, add_general_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument('--task', type=str, default='traffic_state_pred', help='the name of task')
    parser.add_argument('--model', type=str, default='CKGGNN', help='the name of model')
    parser.add_argument('--dataset', type=str, default='speed_test_data', help='the name of dataset')
    parser.add_argument('--config_file', type=str, default=None, help='the file name of config file')
    parser.add_argument('--saved_model', type=str2bool, default=True, help='whether save the trained model')
    parser.add_argument('--train', type=str2bool, default=True, help='re-train model if the model is trained before')
    parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--atten_type', type=str, default='head')
    parser.add_argument('--head_type', type=str, default='FeaSeqPlus')
    parser.add_argument('--head_num_seq', type=int, default=10)
    parser.add_argument('--head_num_fea', type=int, default=4)
    parser.add_argument('--wd_set', type=float, default=0.00001)
    parser.add_argument('--fuse_method', type=str, default='add')
    parser.add_argument('--kg_weight', type=str, default='times')

    # dcrnn
    parser.add_argument('--set_cl_decay_steps', type=int, default=2000)
    parser.add_argument('--set_max_diffusion_step', type=int, default=2)
    parser.add_argument('--set_num_rnn_layers', type=int, default=2)
    parser.add_argument('--set_rnn_units', type=int, default=64)
    parser.add_argument('--set_bidir_adj_mx', type=str2bool, default=False)

    # spatial kg parameter
    parser.add_argument('--spat_model_used', type=str, default='ComplEx')
    parser.add_argument('--spat_pickle_file', type=str, default='./kg_data/kg_spatial_pickle_dict.pickle')
    parser.add_argument('--spat_cont_used', type=str, default='road-poi-land')
    parser.add_argument('--spat_attr_used', type=str, default='road-poi-land')
    parser.add_argument('--spat_link_used', type=int, default=6)
    parser.add_argument('--spat_buffer_used', type=str, default='10-100')
    parser.add_argument('--spat_link_attr', type=int, default=6)
    parser.add_argument('--spat_buffer_attr', type=str, default='10-100')

    # temporal kg parameter
    parser.add_argument('--temp_model_used', type=str, default='KG2E')
    parser.add_argument('--temp_pickle_file', type=str, default='./kg_data/kg_temporal_pickle_dict.pickle')
    parser.add_argument('--temp_cont_used', type=str, default='time-jam-weather')
    parser.add_argument('--temp_attr_used', type=str, default='time-jam-weather')
    parser.add_argument('--temp_time_used', type=int, default=6, help='1-12')
    parser.add_argument('--temp_link_used', type=str, default='hour-day-week')
    parser.add_argument('--temp_time_attr', type=int, default=6, help='1-12')
    parser.add_argument('--temp_link_attr', type=str, default='hour')  # single day
    parser.add_argument('--temp_datetime', type=str, default='2022-03-23T08:05:00Z')

    # kg embedding paramter
    parser.add_argument('--kg_context', type=str, default='both', help='spat/temp/both')
    parser.add_argument('--kg_switch', type=str2bool, default=True, help='decide whether use knowledge graphs')
    parser.add_argument('--kg_embed_dim', type=int, default=30)
    parser.add_argument('--kg_epochs_num', type=int, default=100)

    # add other parameters
    add_general_args(parser)
    # parser parameter
    args = parser.parse_args()
    args.task = 'traffic_state_pred'
    args.dataset = 'speed_test_data'
    args.model = 'CKGGNN'
    args.batch_size = 16
    args.max_epoch = 500
    dict_args = vars(args)

    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
        val is not None}
    run_model(task=args.task, model_name=args.model, dataset_name=args.dataset,
              config_file=args.config_file, saved_model=args.saved_model,
              train=args.train, other_args=other_args)
