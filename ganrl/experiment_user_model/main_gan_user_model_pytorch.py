#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：albert time:2019/8/8
import datetime
import numpy as np
import os
import threading

from ganrl.common.cmd_args import cmd_args
from ganrl.experiment_user_model.data_utils import Dataset
from ganrl.experiment_user_model.gan_user_model_pytorch import GanNet


if __name__ == '__main__':

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start" % log_time)

    dataset = Dataset(cmd_args)

    if cmd_args.resplit:
        dataset.random_split_user()

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, load data completed" % log_time)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start to construct graph" % log_time)

    # if cmd_args.user_model == 'LSTM':
    #     user_model = UserModelLSTM(dataset.f_dim, cmd_args)
    # elif cmd_args.user_model == 'PW':
    #     user_model = UserModelPW(dataset.f_dim, cmd_args)
    # else:
    #     print('using LSTM user model instead.')
    #     user_model = UserModelLSTM(dataset.f_dim, cmd_args)

    if cmd_args.user_model == 'PW':
        user_model = GanNet(dataset.f_dim, cmd_args)



    # prepare validation data
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start prepare vali data" % log_time)

    if cmd_args.user_model == 'PW':
        vali_thread_user, click_2d_vali, disp_2d_vali, \
        feature_vali, sec_cnt_vali, tril_ind_vali, tril_value_ind_vali, disp_2d_split_sec_vali, \
        news_cnt_short_vali, click_sub_index_2d_vali, feature_clicked_vali = dataset.prepare_validation_data(cmd_args.num_thread, dataset.vali_user)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, prepare validation data, completed" % log_time)

    best_metric = [100000.0, 0.0, 0.0]


    for i in range(cmd_args.num_itrs):
        # training_start_point = (i * cmd_args.batch_size) % (len(dataset.train_user))
        # training_user = dataset.train_user[training_start_point: min(training_start_point + cmd_args.batch_size, len(dataset.train_user))]

        training_user = np.random.choice(dataset.train_user, cmd_args.batch_size, replace=False)


        if cmd_args.user_model == 'PW':
            click_2d, disp_2d, feature_tr, sec_cnt, tril_ind, tril_value_ind, disp_2d_split_sect, \
            news_cnt_sht, click_2d_subind, feature_clicked_tr = dataset.data_process_for_placeholder(training_user)

            a = user_model(sec_cnt, news_cnt_sht, tril_value_ind, tril_ind,
                           feature_clicked_tr, disp_2d_split_sect,
                           feature_tr)
            print(' ')



