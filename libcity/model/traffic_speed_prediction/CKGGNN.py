import os, re
import pickle
import numpy as np
from logging import getLogger
from datetime import datetime
import scipy.sparse as sp
from scipy.sparse import linalg
import torch
import torch.nn as nn
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def calculate_normalized_laplacian(adj):
    """
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2

    Args:
        adj: adj matrix

    Returns:
        np.ndarray: L
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GCONV(nn.Module):
    def __init__(self, num_nodes, max_diffusion_step, supports, device, input_dim, hid_dim, output_dim, bias_start=0.0):
        super().__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._supports = supports
        self._device = device
        self._num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Ks
        self._output_dim = output_dim
        input_size = input_dim + hid_dim
        shape = (input_size * self._num_matrices, self._output_dim)
        self.weight = torch.nn.Parameter(torch.empty(*shape, device=self._device))
        self.biases = torch.nn.Parameter(torch.empty(self._output_dim, device=self._device))
        torch.nn.init.xavier_normal_(self.weight)
        torch.nn.init.constant_(self.biases, bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, inputs, state):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        # (batch_size, num_nodes, total_arg_size(input_dim+state_dim))
        input_size = inputs_and_state.size(2)  # =total_arg_size

        x = inputs_and_state
        # T0=I x0=T0*x=x
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)  # (1, num_nodes, total_arg_size * batch_size)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                # T1=L x1=T1*x=L*x
                x1 = torch.sparse.mm(support, x0)  # supports: n*n; x0: n*(total_arg_size * batch_size)
                x = self._concat(x, x1)  # (2, num_nodes, total_arg_size * batch_size)
                for k in range(2, self._max_diffusion_step + 1):
                    # T2=2LT1-T0=2L^2-1 x2=T2*x=2L^2x-x=2L*x1-x0...
                    # T3=2LT2-T1=2L(2L^2-1)-L x3=2L*x2-x1...
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)  # (3, num_nodes, total_arg_size * batch_size)
                    x1, x0 = x2, x1
        # x.shape (Ks, num_nodes, total_arg_size * batch_size)
        # Ks = len(supports) * self._max_diffusion_step + 1

        x = torch.reshape(x, shape=[self._num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, num_matrices)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * self._num_matrices])

        x = torch.matmul(x, self.weight)  # (batch_size * self._num_nodes, self._output_dim)
        x += self.biases
        # Reshape res back to 2D: (batch_size * num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * self._output_dim])


class FC(nn.Module):
    def __init__(self, num_nodes, device, input_dim, hid_dim, output_dim, bias_start=0.0):
        super().__init__()
        self._num_nodes = num_nodes
        self._device = device
        self._output_dim = output_dim
        input_size = input_dim + hid_dim
        shape = (input_size, self._output_dim)
        self.weight = torch.nn.Parameter(torch.empty(*shape, device=self._device))
        self.biases = torch.nn.Parameter(torch.empty(self._output_dim, device=self._device))
        torch.nn.init.xavier_normal_(self.weight)
        torch.nn.init.constant_(self.biases, bias_start)

    def forward(self, inputs, state):
        batch_size = inputs.shape[0]
        # Reshape input and state to (batch_size * self._num_nodes, input_dim/state_dim)
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        # (batch_size * self._num_nodes, input_size(input_dim+state_dim))
        value = torch.sigmoid(torch.matmul(inputs_and_state, self.weight))
        # (batch_size * self._num_nodes, self._output_dim)
        value += self.biases
        # Reshape res back to 2D: (batch_size * num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(value, [batch_size, self._num_nodes * self._output_dim])


class DCGRUCell(nn.Module):
    def __init__(self, input_dim, num_units, adj_mx, max_diffusion_step, num_nodes, device, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True):
        """

        Args:
            input_dim:
            num_units:
            adj_mx:
            max_diffusion_step:
            num_nodes:
            device:
            nonlinearity:
            filter_type: "laplacian", "random_walk", "dual_random_walk"
            use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._device = device
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru

        supports = []
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
            supports.append(calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support, self._device))

        if self._use_gc_for_ru:
            self._fn = GCONV(self._num_nodes, self._max_diffusion_step, self._supports, self._device,
                             input_dim=input_dim, hid_dim=self._num_units, output_dim=2*self._num_units, bias_start=1.0)
        else:
            self._fn = FC(self._num_nodes, self._device, input_dim=input_dim,
                          hid_dim=self._num_units, output_dim=2*self._num_units, bias_start=1.0)
        self._gconv = GCONV(self._num_nodes, self._max_diffusion_step, self._supports, self._device,
                            input_dim=input_dim, hid_dim=self._num_units, output_dim=self._num_units, bias_start=0.0)

    @staticmethod
    def _build_sparse_matrix(lap, device):
        lap = lap.tocoo()
        indices = np.column_stack((lap.row, lap.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        lap = torch.sparse_coo_tensor(indices.T, lap.data, lap.shape, device=device)
        return lap

    def forward(self, inputs, hx):
        """
        Gated recurrent unit (GRU) with Graph Convolution.

        Args:
            inputs: (B, num_nodes * input_dim)
            hx: (B, num_nodes * rnn_units)

        Returns:
            torch.tensor: shape (B, num_nodes * rnn_units)
        """
        output_size = 2 * self._num_units
        value = torch.sigmoid(self._fn(inputs, hx))  # (batch_size, num_nodes * output_size)
        value = torch.reshape(value, (-1, self._num_nodes, output_size))    # (batch_size, num_nodes, output_size)

        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))  # (batch_size, num_nodes * _num_units)
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))  # (batch_size, num_nodes * _num_units)

        c = self._gconv(inputs, r * hx)  # (batch_size, num_nodes * _num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state  # (batch_size, num_nodes * _num_units)


class Seq2SeqAttrs:
    def __init__(self, config, adj_mx):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(config.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(config.get('cl_decay_steps', 1000))
        self.filter_type = config.get('filter_type', 'laplacian')
        self.num_nodes = int(config.get('num_nodes', 1))
        self.num_rnn_layers = int(config.get('num_rnn_layers', 2))
        self.rnn_units = int(config.get('rnn_units', 64))
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.input_dim = config.get('feature_dim', 1)
        self.device = config.get('device', torch.device('cpu'))


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(DCGRUCell(self.input_dim, self.rnn_units, adj_mx, self.max_diffusion_step,
                                           self.num_nodes, self.device, filter_type=self.filter_type))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(DCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type))

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        Args:
            inputs: shape (batch_size, self.num_nodes * self.input_dim)
            hidden_state: (num_layers, batch_size, self.hidden_state_size),
                optional, zeros if not provided, hidden_state_size = num_nodes * rnn_units

        Returns:
            tuple: tuple contains:
                output: shape (batch_size, self.hidden_state_size) \n
                hidden_state: shape (num_layers, batch_size, self.hidden_state_size) \n
                (lower indices mean lower layers)

        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size), device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class EncoderModel_goal(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.input_dim = 1  # config.get('feature_dim', 1)
        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(DCGRUCell(self.input_dim, self.rnn_units, adj_mx, self.max_diffusion_step,
                                           self.num_nodes, self.device, filter_type=self.filter_type))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(DCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type))

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        Args:
            inputs: shape (batch_size, self.num_nodes * self.input_dim)
            hidden_state: (num_layers, batch_size, self.hidden_state_size),
                optional, zeros if not provided, hidden_state_size = num_nodes * rnn_units

        Returns:
            tuple: tuple contains:
                output: shape (batch_size, self.hidden_state_size) \n
                hidden_state: shape (num_layers, batch_size, self.hidden_state_size) \n
                (lower indices mean lower layers)

        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size), device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.output_dim = config.get('output_dim', 1)
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(DCGRUCell(self.output_dim, self.rnn_units, adj_mx, self.max_diffusion_step,
                                           self.num_nodes, self.device, filter_type=self.filter_type))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(DCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type))

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        Args:
            inputs:  shape (batch_size, self.num_nodes * self.output_dim)
            hidden_state: (num_layers, batch_size, self.hidden_state_size),
                optional, zeros if not provided, hidden_state_size = num_nodes * rnn_units

        Returns:
            tuple: tuple contains:
                output: shape (batch_size, self.num_nodes * self.output_dim) \n
                hidden_state: shape (num_layers, batch_size, self.hidden_state_size) \n
                (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)
        return output, torch.stack(hidden_states)


class DecoderModel_goal(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.output_dim = config.get('output_dim', 1)
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(DCGRUCell(self.output_dim, self.rnn_units, adj_mx, self.max_diffusion_step,
                                           self.num_nodes, self.device, filter_type=self.filter_type))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(DCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type))

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        Args:
            inputs:  shape (batch_size, self.num_nodes * self.output_dim)
            hidden_state: (num_layers, batch_size, self.hidden_state_size),
                optional, zeros if not provided, hidden_state_size = num_nodes * rnn_units

        Returns:
            tuple: tuple contains:
                output: shape (batch_size, self.num_nodes * self.output_dim) \n
                hidden_state: shape (num_layers, batch_size, self.hidden_state_size) \n
                (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)
        return output, torch.stack(hidden_states)


class CKGGNN(AbstractTrafficStateModel, Seq2SeqAttrs):
    def __init__(self, config, data_feature):
        # DCRNN
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self._config1 = config
        self.embed_dim = config.get('kg_embed_dim')
        self.time_intervals = config.get('time_intervals')

        # attention
        self.atten_type = config.get('atten_type', 'head')
        self.head_type = config.get('head_type', 'FeaSeqPlus')

        # feature dim
        self.pure_feature_dim = 1
        self.fuse_feature_dim = 30
        self.ctx_feature_dim = 0
        self.ctx_feature_dim_pre = 0
        if self.atten_type == 'head':
            self.ctx_feature_dim = self.embed_dim * (1+3+6 + 1+3+3)
            self.ctx_feature_dim_pre = self.ctx_feature_dim
        self.feature_dim = self.ctx_feature_dim + 1

        config['num_nodes'] = self.num_nodes
        config['feature_dim'] = self.feature_dim
        self.output_dim = data_feature.get('output_dim', 1)
        super().__init__(config, data_feature)
        Seq2SeqAttrs.__init__(self, config, self.adj_mx)
        self.encoder_model = EncoderModel(config, self.adj_mx)
        self.decoder_model = DecoderModel(config, self.adj_mx)
        self.encoder_model_goal = EncoderModel_goal(config, self.adj_mx)
        self.decoder_model_goal = DecoderModel_goal(config, self.adj_mx)

        self.use_curriculum_learning = config.get('use_curriculum_learning', False)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        # context
        self.kg_context = config.get('kg_context')
        self.spat_model_used = config.get('spat_model_used')
        self.spat_attr_used = config.get('spat_attr_used')
        self.spat_link_attr = config.get('spat_link_attr')
        self.spat_buffer_attr = config.get('spat_buffer_attr')
        self.temp_model_used = config.get('temp_model_used')
        self.temp_attr_used = config.get('temp_attr_used')
        self.temp_time_attr = config.get('temp_time_attr')
        self.temp_link_attr = config.get('temp_link_attr')

        # others
        self.kg_weight = config.get('kg_weight', 'times')
        self.fc_concat = nn.Linear(self.embed_dim + 1, self.embed_dim)
        self.fc_fuse = nn.Linear(self.ctx_feature_dim, self.fuse_feature_dim).to(self.device)
        self.out_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1))
        self.out_fc = nn.Linear(2, 1).to(self.device)

        # sequence MH level
        self.attn_head_seq = config.get('head_num_seq', 10)
        self.self_attn_seq = nn.MultiheadAttention(embed_dim=self.ctx_feature_dim_pre, num_heads=self.attn_head_seq)
        # feature MH level
        self.attn_head_fea = config.get('head_num_fea', 4)
        self.reduced_dim_per = 1
        self.feature_dim_per = self.embed_dim
        self.ctx_feature_num = 1 + 3 + 6 + 1 + 3 + 3
        self.feature_ln = [nn.Linear(self.feature_dim_per, self.reduced_dim_per).to(self.device) for _ in range(self.ctx_feature_num)]
        self.self_attn_fea = nn.MultiheadAttention(embed_dim=self.input_window, num_heads=self.attn_head_fea)
        self._config2 = config

    def cal_ent_via_rel(self, model, ent, rel, weight):
        ent_change = []
        if self.kg_weight == 'add':
            if model == 'ComplEx':
                ent_change = np.dot(np.diag(rel), ent) + weight
                ent_change = np.real(ent_change)
            elif model == 'KG2E':
                ent_change = ent - rel + weight
            elif model == 'AutoSF':
                ent_change = np.dot(np.diag(rel), ent) + weight
            else:
                self._logger.info('[ERROR]model-{} was not used in embeddings'.format(model))
        elif self.kg_weight == 'times':
            if model == 'ComplEx':
                ent_change = weight * np.dot(np.diag(rel), ent)
                ent_change = np.real(ent_change)
            elif model == 'KG2E':
                ent_change = weight * (ent - rel)
            elif model == 'AutoSF':
                ent_change = weight * np.dot(np.diag(rel), ent)
            else:
                self._logger.info('[ERROR]model-{} was not used in embeddings'.format(model))
        elif self.kg_weight == 'concat':
            if model == 'ComplEx':
                ent_change = np.dot(np.diag(rel), ent)
                ent_change = np.real(ent_change)
                ent_change = np.concatenate([np.array([weight]), ent_change])
            elif model == 'KG2E':
                ent_change = ent - rel
                ent_change = np.concatenate([np.array([weight]), ent_change])
            elif model == 'AutoSF':
                ent_change = np.dot(np.diag(rel), ent)
                ent_change = np.concatenate([np.array([weight]), ent_change])
            else:
                self._logger.info('[ERROR]model-{} was not used in embeddings'.format(model))
            ent_change_tensor = torch.tensor(ent_change).float().to(self.device)
            ent_change_tensor = self.fc_concat(ent_change_tensor)
            ent_change = ent_change_tensor.cpu().detach().numpy()
        return np.array(ent_change)

    def obtain_spat_kge_final(self, x_goal, x_auxi, dict_kge):
        spat_ent_kge, spat_rel_kge = dict_kge['spat_ent_kge'], dict_kge['spat_rel_kge']
        spat_ent_label, spat_rel_label = dict_kge['spat_ent_label'], dict_kge['spat_rel_label']

        # sub_kg for spat
        subdict_spat = dict_kge['sub_spat']  # {road_id: {road/poi/land/link: [fact]}}
        spat_attr_used_list = self.spat_attr_used.split('-')

        # dict_final_spat
        subdict_spat_kge = {}
        # generate spat embeddings
        for _dim3 in range(x_goal.shape[2]):  # num_nodes
            road_str = 'road_' + str(int(x_auxi[0, 0, _dim3, 3]))
            # spat embedding, 3*embed: 1-itself, 2-cont, 3-link
            spat_embed_1 = np.zeros(self.embed_dim, dtype=np.float)
            spat_embed_2 = np.zeros(self.embed_dim*3, dtype=np.float)
            embed2_road = np.zeros(self.embed_dim, dtype=np.float)
            embed2_poi = np.zeros(self.embed_dim, dtype=np.float)
            embed2_land = np.zeros(self.embed_dim, dtype=np.float)
            spat_embed_3 = np.zeros(self.embed_dim*6, dtype=np.float)
            _subkg_fact_road, _subkg_fact_poi, _subkg_fact_land = [], [], []
            if 'road' in spat_attr_used_list:
                _subkg_fact_road.extend(subdict_spat[road_str]['road'])
            if 'poi' in spat_attr_used_list:
                _subkg_fact_poi.extend(subdict_spat[road_str]['poi'])
            if 'land' in spat_attr_used_list:
                _subkg_fact_land.extend(subdict_spat[road_str]['land'])
            # embed 1
            for _fact in _subkg_fact_road:
                if _fact[0] == road_str and _fact[1] == 'self_ffspeed':  # embed-1
                    _self_ffspeed = float(_fact[3])
                    if self.kg_context == 'temp': _self_ffspeed = 0.0
                    if self.spat_model_used == 'ComplEx':
                        spat_embed_1 = np.real(np.array(_self_ffspeed * spat_ent_kge[spat_ent_label[_fact[0]], :]))
                    else:
                        spat_embed_1 = np.array(_self_ffspeed * spat_ent_kge[spat_ent_label[_fact[0]], :])
            # embed 2
            for _fact in _subkg_fact_road:  # road, [road_id, intersectWithRoad[Order], road_id]
                _weight = float(_fact[3])
                if self.kg_context == 'temp': _weight = 0.0
                if _fact[0] == road_str:
                    if _fact[1] != 'self_ffspeed':
                        embed2_road += self.cal_ent_via_rel(self.spat_model_used, spat_ent_kge[spat_ent_label[_fact[2]], :],
                                                            spat_rel_kge[spat_rel_label[_fact[1]], :], _weight)
                else:
                    self._logger.info('[ERROR-road]One fact doesnot find its road_id with fact.{}'.format(_fact))
                    exit()
            for _fact in _subkg_fact_poi:  # poi, [poi_type, locateInBuffer[Dist], road_id]
                _weight = float(_fact[3])
                if self.kg_context == 'temp': _weight = 0.0
                if _fact[2] == road_str:
                    embed2_poi += self.cal_ent_via_rel(self.spat_model_used, spat_ent_kge[spat_ent_label[_fact[0]], :],
                                                       spat_rel_kge[spat_rel_label[_fact[1]], :], _weight)
                else:
                    self._logger.info('[ERROR-poi]One fact doesnot find its road_id with fact.{}'.format(_fact))
                    exit()
            for _fact in _subkg_fact_land:  # land, [land_type, intersectWithBuffer[Dist], road_id]
                _weight = float(_fact[3])
                if self.kg_context == 'temp': _weight = 0.0
                if _fact[2] == road_str:
                    embed2_land += self.cal_ent_via_rel(self.spat_model_used, spat_ent_kge[spat_ent_label[_fact[0]], :],
                                                        spat_rel_kge[spat_rel_label[_fact[1]], :], _weight)
                else:
                    self._logger.info('[ERROR-land]One fact doesnot find its road_id with fact.{}'.format(_fact))
                    exit()
            spat_embed_2[: self.embed_dim] = embed2_road
            spat_embed_2[self.embed_dim: 2 * self.embed_dim] = embed2_poi
            spat_embed_2[2 * self.embed_dim: 3 * self.embed_dim] = embed2_land
            # embed 3
            if self.spat_link_attr > 0:
                for _fact in subdict_spat[road_str]['link']:  # link, [road_id, spatiallyLinkDegree[Degree], road_id]
                    _weight = float(_fact[3])
                    if self.kg_context == 'temp': _weight = 0.0
                    if _fact[0] == road_str:
                        link_degree = int(re.search(r'\d+', _fact[1]).group())
                        embed3_link = self.cal_ent_via_rel(self.spat_model_used, spat_ent_kge[spat_ent_label[_fact[2]], :],
                                                           spat_rel_kge[spat_rel_label[_fact[1]], :], _weight)
                        spat_embed_3[(link_degree-1)*self.embed_dim: link_degree*self.embed_dim] += embed3_link
                    else:
                        self._logger.info('[ERROR-link]One fact doesnot find its road_id with fact.{}'.format(_fact))
                        exit()
            final_embed = np.concatenate((spat_embed_1, spat_embed_2, spat_embed_3), axis=0)
            subdict_spat_kge[road_str] = final_embed
        return subdict_spat_kge

    def convert2datetime(self, x_auxi_slice):
        part1, part2, total_length = map(int, x_auxi_slice)
        long_num = part1 * 10 ** (total_length - len(str(part1))) + part2
        return datetime.fromtimestamp(long_num)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps

        Args:
            inputs: shape (input_window, batch_size, num_sensor * input_dim)

        Returns:
            torch.tensor: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.input_window):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)
            # encoder_hidden_state: encoder的多层GRU的全部的隐层 (num_layers, batch_size, self.hidden_state_size)

        return encoder_hidden_state  # 最后一个隐状态

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass

        Args:
            encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
            labels:  (self.output_window, batch_size, self.num_nodes * self.output_dim)
                [optional, not exist for inference]
            batches_seen: global step [optional, not exist for inference]

        Returns:
            torch.tensor: (self.output_window, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.output_dim), device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []
        for t in range(self.output_window):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output     # (batch_size, self.num_nodes * self.output_dim)
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]  # (batch_size, self.num_nodes * self.output_dim)
        outputs = torch.stack(outputs)
        return outputs

    def encoder_goal(self, inputs):
        """
        encoder forward pass on t time steps

        Args:
            inputs: shape (input_window, batch_size, num_sensor * input_dim)

        Returns:
            torch.tensor: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.input_window):
            _, encoder_hidden_state = self.encoder_model_goal(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder_goal(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass

        Args:
            encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
            labels:  (self.output_window, batch_size, self.num_nodes * self.output_dim)
                [optional, not exist for inference]
            batches_seen: global step [optional, not exist for inference]

        Returns:
            torch.tensor: (self.output_window, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.output_dim), device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []
        for t in range(self.output_window):
            decoder_output, decoder_hidden_state = self.decoder_model_goal(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output     # (batch_size, self.num_nodes * self.output_dim)
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]  # (batch_size, self.num_nodes * self.output_dim)
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, batch, batches_seen=None, dict_kge=None):
        """
        seq2seq forward pass

        Args:
            batch: a batch of input,
                batch['X']: shape (batch_size, input_window, num_nodes, input_dim) \n
                batch['y']: shape (batch_size, output_window, num_nodes, output_dim) \n
            batches_seen: batches seen till now

        Returns:
            torch.tensor: (batch_size, self.output_window, self.num_nodes, self.output_dim)
        """

        # dyna variables
        x_auxi = batch['X_auxi']  # (batch_size, input_window, num_nodes, feature_dim(4))
        x_goal = batch['X_goal']  # (batch_size, input_window, num_nodes, feature_dim(1))
        x_spat = np.zeros((x_goal.shape[0], x_goal.shape[1], x_goal.shape[2], self.embed_dim*(1+3+6)))
        x_temp = np.zeros((x_goal.shape[0], x_goal.shape[1], x_goal.shape[2], self.embed_dim*(1+3+3)))

        # sub_kg and embed for spat
        subdict_spat_kge = dict_kge['sub_spat_emd']
        if len(subdict_spat_kge) == 0:
            subdict_spat_kge = self.obtain_spat_kge_final(x_goal, x_auxi, dict_kge)
            dict_kge['sub_spat_emd'] = subdict_spat_kge

        # sub_kg and embed for temp: each time slot corresponds a subkg
        if self.kg_context != 'spat':
            for _dim1 in range(x_goal.shape[0]):  # batch_size
                if _dim1 == 0:
                    for _dim2 in range(x_goal.shape[1]):  # input_window
                        temp_datetime = self.convert2datetime(x_auxi[_dim1, _dim2, 0, 0:3])
                        subdict_temp_kge = dict_kge['sub_temp_emd'][temp_datetime]
                        road_strs = ['road_' + str(int(i)) for i in x_auxi[_dim1, _dim2, :, 3]]
                        x_spat[_dim1, _dim2, :, :] = [subdict_spat_kge[road] for road in road_strs]
                        x_temp[_dim1, _dim2, :, :] = [subdict_temp_kge[road] for road in road_strs]
                else:
                    # Shift previous values to the left
                    x_spat[_dim1, :-1, :, :] = x_spat[_dim1 - 1, 1:, :, :]
                    x_temp[_dim1, :-1, :, :] = x_temp[_dim1 - 1, 1:, :, :]
                    # Calculate new values for the rightmost position in the window
                    new_dim2 = x_goal.shape[1] - 1
                    temp_datetime = self.convert2datetime(x_auxi[_dim1, new_dim2, 0, 0:3])
                    subdict_temp_kge = dict_kge['sub_temp_emd'][temp_datetime]

                    road_strs = ['road_' + str(int(i)) for i in x_auxi[_dim1, new_dim2, :, 3]]
                    x_spat[_dim1, -1, :, :] = [subdict_spat_kge[road] for road in road_strs]
                    x_temp[_dim1, -1, :, :] = [subdict_temp_kge[road] for road in road_strs]
        else:  # self.kg_context == 'spat'
            road_strs = ['road_' + str(int(i)) for i in x_auxi[0, 0, :, 3]]
            common_x_spat_value = [subdict_spat_kge[road] for road in road_strs]  # x_spat[0, 0, :, :]
            # Use broadcasting to assign the value to all positions of _dim1 and _dim2
            x_spat[:, :, :, :] = common_x_spat_value

        # attention
        attn_output = []
        x_spat = x_spat.astype(np.float32)
        x_spat = torch.from_numpy(x_spat).to(x_goal.device)
        x_temp = x_temp.astype(np.float32)
        x_temp = torch.from_numpy(x_temp).to(x_goal.device)

        if self.atten_type == 'head':
            x_ctx = torch.cat((x_spat, x_temp), dim=-1)
            x_ctx = x_ctx.permute(1, 0, 2, 3)  # (input_window, batch_size, num_nodes, feature_dim)
            if self.head_type == 'FeaSeqPlus':
                # feature level
                reduced_features = []
                for i in range(self.ctx_feature_num):
                    start_idx = i * self.feature_dim_per
                    end_idx = (i + 1) * self.feature_dim_per
                    feature_per = x_ctx[:, :, :, start_idx:end_idx].reshape(-1, self.feature_dim_per)
                    reduced_feature = self.feature_ln[i](feature_per)
                    reduced_feature = reduced_feature.reshape(x_ctx.shape[0], x_ctx.shape[1], x_ctx.shape[2], self.reduced_dim_per)
                    reduced_features.append(reduced_feature)
                x_reduced = torch.cat(reduced_features, dim=3).to(x_goal.device)
                x_reshaped = x_reduced.permute(3, 1, 2, 0)
                x_reshaped = x_reshaped.reshape(x_reduced.shape[3], x_reduced.shape[1] * x_reduced.shape[2], x_reduced.shape[0])
                attn_output_fea, attn_weights_fea = self.self_attn_fea(x_reshaped, x_reshaped, x_reshaped, need_weights=True, average_attn_weights=False)
                attn_weights_mean = attn_weights_fea.mean(dim=1)
                attn_output_new = torch.zeros_like(x_ctx).to(x_goal.device)
                for i in range(self.ctx_feature_num):
                    start_idx = i * self.feature_dim_per
                    end_idx = (i + 1) * self.feature_dim_per
                    feature_per = x_ctx[:, :, :, start_idx:end_idx].reshape(-1, self.feature_dim_per)
                    attn_weight_per = attn_weights_mean[:, i, i].unsqueeze(1).unsqueeze(-1)
                    attn_weight_per = attn_weight_per.expand(-1, self.input_window, -1).reshape(-1, 1)
                    weighted_feature = feature_per * attn_weight_per
                    attn_output_new[:, :, :, start_idx:end_idx] = weighted_feature.reshape(x_ctx.shape[0], x_ctx.shape[1], x_ctx.shape[2], self.feature_dim_per)
                attn_output_fea = attn_output_new
                # sequence level
                x_combined = x_ctx.reshape(x_ctx.shape[0], x_ctx.shape[1] * x_ctx.shape[2], x_ctx.shape[3])
                attn_mask = torch.triu(torch.ones(x_combined.shape[0], x_combined.shape[0]), diagonal=1).bool()
                attn_mask = attn_mask.to(x_goal.device)
                attn_output_seq, attn_weights_seq = self.self_attn_seq(x_combined, x_combined, x_combined, attn_mask=attn_mask, need_weights=True)
                attn_output_seq = attn_output_seq.reshape(x_ctx.shape[0], x_ctx.shape[1], x_ctx.shape[2], -1)
                # output
                attn_output = attn_output_fea + attn_output_seq
            else:
                self._logger.info('NO {} in MultiHeadAttention'.format(self.head_type))
                exit()
            attn_output = attn_output.permute(1, 0, 2, 3)  # (batch_size, input_window, num_nodes, feature_dim)

        # release resource
        del x_goal, x_auxi, x_spat, x_temp
        torch.cuda.empty_cache()

        # dcrnn: y
        labels = batch['y_goal']
        batch_size, _, num_nodes, __ = labels.shape
        if labels is not None:
            labels = labels.permute(1, 0, 2, 3)  # (output_window, batch_size, num_nodes, output_dim)
            labels = labels[..., :self.output_dim].reshape(
                self.output_window, batch_size, num_nodes * self.output_dim).to(self.device)
            self._logger.debug("y: {}".format(labels.size()))

        # dcrnn: x
        inputs_context = torch.cat((batch['X_goal'], attn_output), dim=-1)
        batch_size, _, num_nodes, input_dim = inputs_context.shape
        inputs_context = inputs_context.permute(1, 0, 2, 3)  # (input_window, batch_size, num_nodes, input_dim)
        inputs_context = inputs_context.reshape(self.input_window, batch_size, num_nodes * input_dim).to(self.device)
        self._logger.debug("X: {}".format(inputs_context.size()))  # (input_window, batch_size, num_nodes * input_dim)
        encoder_hidden_state_ctx = self.encoder(inputs_context)
        self._logger.debug("Encoder_context complete")
        # (self.output_window, batch_size, self.num_nodes * self.output_dim)
        outputs_context = self.decoder(encoder_hidden_state_ctx, labels, batches_seen=batches_seen)
        self._logger.debug("Decoder_context complete")
        if batches_seen == 0:
            self._logger.info("Total trainable parameters {}".format(count_parameters(self)))
        outputs_context = outputs_context.view(self.output_window, batch_size, self.num_nodes, self.output_dim).permute(1, 0, 2, 3)
        return outputs_context

    def calculate_loss(self, batch, batches_seen=None, dict_kge=None):
        y_true = batch['y_goal']
        y_predicted = self.predict(batch, batches_seen, dict_kge)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch, batches_seen=None, dict_kge=None):
        return self.forward(batch, batches_seen, dict_kge)
