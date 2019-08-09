#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：albert time:2019/8/8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import random

def segment_sum(data, segment_ids):
    """
    Analogous to tf.segment_sum (https://www.tensorflow.org/api_docs/python/tf/math/segment_sum).

    :param data: A pytorch tensor of the data for segmented summation.
    :param segment_ids: A 1-D tensor containing the indices for the segmentation.
    :return: a tensor of the same type as data containing the results of the segmented summation.
    """
    # 判断segment_id是否有序
    if not all(segment_ids[i] <= segment_ids[i + 1] for i in range(len(segment_ids) - 1)):
        raise AssertionError("elements of segment_ids must be sorted")
    # index必须是一维张量
    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have be a 1-D tensor")
    # data要和segment_id长度相同
    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError("segment_ids should be the same size as dimension 0 of input.")

    # t_grp = {}
    # idx = 0
    # for i, s_id in enumerate(segment_ids):
    #     s_id = s_id.item()
    #     if s_id in t_grp:
    #         t_grp[s_id] = t_grp[s_id] + data[idx]
    #     else:
    #         t_grp[s_id] = data[idx]
    #     idx = i + 1
    #
    # lst = list(t_grp.values())
    # tensor = torch.stack(lst)

    # num_segments，segment的数量
    num_segments = len(torch.unique(segment_ids))
    return unsorted_segment_sum(data, segment_ids, num_segments)

# 利用scatter_add：先建立一个全零矩阵，然后将每个tau有值的地方填进去,然后再加和
def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape).scatter_add(0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor


class mlp(nn.Module):
    def __init__(self, hidden_dims, output_dim, activation, sd, x_dim, act_last=False,):
        super(mlp, self).__init__()

        self.hidden_dims = tuple(map(int, hidden_dims.split("-")))
        self.output_dim = output_dim
        self.activation = activation
        self.sd = sd
        self.act_last=act_last

        self.layers = []

        h1 = x_dim

        for h in self.hidden_dims:
            layer = nn.Linear(h1, h)

            layer.weight.data.normal_(std=sd)
            self.layers.append(layer)
            h1= h

        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)
        self.output_layer.weight.data.normal_(std=sd)

    def forward(self, x):
        for i in range(len(self.hidden_dims)):
            x = self.layers[i](x)
            x = self.activation(x)

        if self.act_last:
            x = self.activation(self.output_layer(x))
        else:
            x = self.output_layer(x)

        return x

class GanNet(nn.Module):
    def __init__(self, f_dim, args,):
        super(GanNet, self).__init__()
        self.f_dim = f_dim
        self.rnn_hidden = args.rnn_hidden_dim
        self.hidden_dims = args.dims
        self.lr = args.learning_rate

        self.pw_dim = args.pw_dim

        self.band_size = args.pw_band_size

        self.mlp = mlp(self.hidden_dims, 1, F.elu, 1e-3, self.f_dim*self.pw_dim+self.f_dim)


    def forward(self, section_length, item_size, cumsum_tril_value_indices,
                cumsum_tril_indices, Xs_clicked, disp_2d_split_sec_ind,
                disp_current_feature):
        denseshape = torch.Size([section_length, item_size])

        click_history = [[] for _ in range(self.pw_dim)]

        for ii in range(self.pw_dim):
            position_weight = torch.full((self.band_size, ), 0.0001, requires_grad=True)

            # torch.index_select(in, dim, index)作用等于tf.gather，dim=0是因为这里的in只有一维，
            cumsum_tril_value = torch.index_select(position_weight, dim=0, index=torch.LongTensor(cumsum_tril_value_indices))

            cumsum_tril_matrix = torch.sparse.FloatTensor(torch.LongTensor(cumsum_tril_indices).t(), cumsum_tril_value, torch.Size([section_length, section_length]))

            click_history[ii] = torch.sparse.mm(cumsum_tril_matrix, torch.tensor(Xs_clicked))

        concat_history = torch.cat(click_history, dim=1)

        disp_history_feature = torch.index_select(concat_history, dim=0, index=torch.LongTensor(disp_2d_split_sec_ind))

        concat_disp_feature = torch.cat((disp_history_feature, torch.FloatTensor(disp_current_feature)), dim=1)

        # test = torch.randn([870, self.f_dim*self.pw_dim+self.f_dim])
        u_disp = self.mlp(concat_disp_feature)

        exp_u_disp = torch.exp(u_disp)

        sum_exp_disp_ubar_ut = segment_sum(exp_u_disp, torch.LongTensor(disp_2d_split_sec_ind))


        print(" ")






# class UserModelPW(nn.Module):
#     def __init__(self, f_dim, args, max_disp_size=None):
#         self.f_dim = f_dim
#         self.rnn_hidden = args.rnn_hidden_dim
#         self.hidden_dims = args.dims
#         self.lr = args.learning_rate
#         self.max_disp_size = max_disp_size
#
#     def forward(self, x):

