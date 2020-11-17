#!/usr/bin/env python
# -*- coding: utf-8 -*-
import h5py
# import warnings
# with warnings.catch_warnings():
#    warnings.filterwarnings("ignore",category=FutureWarning)
#    import h5py
from data_loader import TurnPredictionDataset
from lstm_model import LSTMPredictor
# from lstm_extra_layer import LSTMPredictor

# from external_network import ExternalNetwork
from torch.nn.utils import clip_grad_norm
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from copy import deepcopy

from os import mkdir
from os.path import exists
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import f1_score, roc_curve, confusion_matrix
import time as t
import pickle
import platform
from sys import argv
import json
from random import randint
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import feature_vars as feat_dicts

# %% data set select
data_set_select = 0  # 0 for maptask, 1 for mahnob, 2 for switchboard
if data_set_select == 0:
    #    train_batch_size = 878
    train_batch_size = 128
    # train_batch_size = 32

    test_batch_size = 1
else:
    train_batch_size = 128
    # train_batch_size = 256
    #    train_batch_size = 830 # change this
    test_batch_size = 1

# %% Batch settings
alpha = 0.99  # smoothing constant
init_std = 0.5
momentum = 0
test_batch_size = 1  # this should stay fixed at 1 when using slow test because the batches are already set in the data loader

# sequence_length = 800
# dropout = 0
prediction_length = 60  # (3 seconds of prediction)
shuffle = True
num_layers = 1
onset_test_flag = True
annotations_dir = './data/extracted_annotations/voice_activity/'

proper_num_args = 2
print('Number of arguments is: ' + str(len(argv)))

####### ling and acoustic ###########

# json_dict = json.loads('{"feature_dict_list": [{"folder_path": "./data/datasets/gemaps_split.hdf5", "features": ["F0semitoneFrom27.5Hz", "jitterLocal", "F1frequency", "F1bandwidth", "F2frequency", "F3frequency", "Loudness", "shimmerLocaldB", "HNRdBACF", "alphaRatio", "hammarbergIndex", "spectralFlux", "slope0-500", "slope500-1500", "F1amplitudeLogRelF0", "F2amplitudeLogRelF0", "F3amplitudeLogRelF0", "mfcc1", "mfcc2", "mfcc3", "mfcc4"], "modality": "acous", "is_h5_file": true, "uses_master_time_rate": false, "time_step_size": 5, "is_irregular": false, "short_name": "gmaps10"}, {"folder_path": "./data/extracted_annotations/words_advanced_50ms_averaged/", "features": ["word"], "modality": "visual", "is_h5_file": false, "uses_master_time_rate": true, "time_step_size": 1, "is_irregular": false, "short_name": "wrd_reg", "title_string": "_word", "embedding": true, "embedding_num": 18359, "embedding_in_dim": 1, "embedding_out_dim": 64, "embedding_use_func": true, "use_glove": false, "glove_embed_table": ""}], "results_dir": "./main/2_Acous_10ms_Ling_50ms/test/", "name_append": "0_2_Acous_10ms_Ling_50ms_m_50_a_50_v_50_lr_01_l2e_0001_l2o_-05_l2m_-05_l2a_0001_l2v_-05_dmo_25_dmi_5_dao_25_dai_25_dvo_0_dvi_25_seq_600", "no_subnets": false, "hidden_nodes_master": 50, "hidden_nodes_acous": 50, "hidden_nodes_visual": 50, "learning_rate": 0.01, "sequence_length": 600, "num_epochs": 1500, "early_stopping": true, "patience": 10, "slow_test": true, "train_list_path": "./data/splits/swb_train_refined.txt", "test_list_path": "./data/splits/swb_test_refined.txt", "use_date_str": false, "freeze_glove_embeddings": false, "grad_clip_bool": false, "l2_dict": {"emb": 0.0001, "out": 1e-05, "master": 1e-05, "acous": 0.0001, "visual": 1e-05}, "dropout_dict": {"master_out": 0.25, "master_in": 0.5, "acous_in": 0.25, "acous_out": 0.25, "visual_in": 0.25, "visual_out": 0.0}}')

####### acoustic only #########

# json_dict = json.loads('{"feature_dict_list": [{"folder_path": "./data/datasets/gemaps_split.hdf5", "features": ["F0semitoneFrom27.5Hz", "jitterLocal", "F1frequency", "F1bandwidth", "F2frequency", "F3frequency", "Loudness", "shimmerLocaldB", "HNRdBACF", "alphaRatio", "hammarbergIndex", "spectralFlux", "slope0-500", "slope500-1500", "F1amplitudeLogRelF0", "F2amplitudeLogRelF0", "F3amplitudeLogRelF0", "mfcc1", "mfcc2", "mfcc3", "mfcc4"], "modality": "acous", "is_h5_file": true, "uses_master_time_rate": false, "time_step_size": 5, "is_irregular": false, "short_name": "gmaps10"}], "results_dir": "./ling_50ms/2_Acous_10ms/test/", "name_append": "0_2_Acous_10ms_m_60_lr_01_l2e_0_l2o_-05_l2m_-05_dmo_5_dmi_5_seq_600", "no_subnets": true, "hidden_nodes_master": 60, "hidden_nodes_acous": 0, "hidden_nodes_visual": 0, "learning_rate": 0.01, "sequence_length": 600, "num_epochs": 1500, "early_stopping": true, "patience": 10, "slow_test": true, "train_list_path": "./data/splits/swb_train_small.txt", "test_list_path": "./data/splits/swb_test_small.txt", "use_date_str": false, "freeze_glove_embeddings": false, "grad_clip_bool": false, "l2_dict": {"emb": 0.0, "out": 1e-05, "master": 1e-05, "acous": 0, "visual": 0}, "dropout_dict": {"master_out": 0.5, "master_in": 0.5, "acous_in": 0, "acous_out": 0, "visual_in": 0, "visual_out": 0.0}}')

######### ling only ############

json_dict = json.loads('{"feature_dict_list": [{"folder_path": "./data/extracted_annotations/words_advanced_50ms_averaged/", "features": ["word"], "modality": "acous", "is_h5_file": false, "uses_master_time_rate": true, "time_step_size": 1, "is_irregular": false, "short_name": "wrd_reg", "title_string": "_word", "embedding": true, "embedding_num": 18359, "embedding_in_dim": 1, "embedding_out_dim": 64, "embedding_use_func": true, "use_glove": false, "glove_embed_table": ""}], "results_dir": "./ling_50ms/3_Ling_50ms/test/", "name_append": "0_3_Ling_50ms_m_60_lr_001_l2e_0001_l2o_0001_l2m_0001_dmo_5_dmi_25_seq_600", "no_subnets": true, "hidden_nodes_master": 60, "hidden_nodes_acous": 0, "hidden_nodes_visual": 0, "learning_rate": 0.001, "sequence_length": 600, "num_epochs": 1500, "early_stopping": true, "patience": 10, "slow_test": true, "train_list_path": "./data/splits/swb_train_small.txt", "test_list_path": "./data/splits/swb_test_small.txt", "use_date_str": false, "freeze_glove_embeddings": false, "grad_clip_bool": false, "l2_dict": {"emb": 0.0001, "out": 0.0001, "master": 0.0001, "acous": 0, "visual": 0}, "dropout_dict": {"master_out": 0.5, "master_in": 0.25, "acous_in": 0, "acous_out": 0, "visual_in": 0, "visual_out": 0.0}}')

############### pos and acous #############################

# json_dict = json.loads('{"feature_dict_list": [{"folder_path": "./data/datasets/gemaps_split.hdf5", "features": ["F0semitoneFrom27.5Hz", "jitterLocal", "F1frequency", "F1bandwidth", "F2frequency", "F3frequency", "Loudness", "shimmerLocaldB", "HNRdBACF", "alphaRatio", "hammarbergIndex", "spectralFlux", "slope0-500", "slope500-1500", "F1amplitudeLogRelF0", "F2amplitudeLogRelF0", "F3amplitudeLogRelF0", "mfcc1", "mfcc2", "mfcc3", "mfcc4"], "modality": "acous", "is_h5_file": true, "uses_master_time_rate": false, "time_step_size": 5, "is_irregular": false, "short_name": "gmaps10"}, {"folder_path": "./data/extracted_annotations/pos_advanced_50ms_averaged/", "features": ["word"], "modality": "visual", "is_h5_file": false, "uses_master_time_rate": true, "time_step_size": 1, "is_irregular": false, "short_name": "wrd_reg", "title_string": "_word", "embedding": true, "embedding_num": 18359, "embedding_in_dim": 1, "embedding_out_dim": 64, "embedding_use_func": true, "use_glove": false, "glove_embed_table": ""}], "results_dir": "./one_sec/pos/test/", "name_append": "0_2_Acous_10ms_Ling_50ms_m_50_a_50_v_50_lr_01_l2e_0001_l2o_-05_l2m_-05_l2a_0001_l2v_-05_dmo_25_dmi_5_dao_25_dai_25_dvo_0_dvi_25_seq_600", "no_subnets": false, "hidden_nodes_master": 50, "hidden_nodes_acous": 50, "hidden_nodes_visual": 50, "learning_rate": 0.01, "sequence_length": 600, "num_epochs": 1500, "early_stopping": true, "patience": 10, "slow_test": true, "train_list_path": "./data/splits/swb_train_refined.txt", "test_list_path": "./data/splits/swb_test_refined.txt", "use_date_str": false, "freeze_glove_embeddings": false, "grad_clip_bool": false, "l2_dict": {"emb": 0.0001, "out": 1e-05, "master": 1e-05, "acous": 0.0001, "visual": 1e-05}, "dropout_dict": {"master_out": 0.25, "master_in": 0.5, "acous_in": 0.25, "acous_out": 0.25, "visual_in": 0.25, "visual_out": 0.0}}')

locals().update(json_dict)

# %% Get OS type and whether to use cuda or not
plat = platform.linux_distribution()[0]
my_node = platform.node()

complete_path = './data/splits/trp_prediction_sets/trp_test.txt'

if (plat == 'arch') | (my_node == 'Matthews-MacBook-Pro.local'):
    print('platform: arch')
else:
    print('platform: ' + str(plat))

use_cuda = torch.cuda.is_available()

print('Use CUDA: ' + str(use_cuda))

if use_cuda:
    #    torch.cuda.device(randint(0,1))
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
    p_memory = True
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor
    p_memory = True

# %% Data loaders
t1 = t.time()

complete_dataset = TurnPredictionDataset(feature_dict_list, annotations_dir, complete_path, sequence_length,
                                      prediction_length, 'test', data_select=data_set_select)

complete_dataloader = DataLoader(complete_dataset, batch_size=1, shuffle=False, num_workers=0,  # previously shuffle = shuffle
                              drop_last=False, pin_memory=p_memory)

feature_size_dict = complete_dataset.get_feature_size_dict()

print('time taken to load data: ' + str(t.time() - t1))

complete_file_list = list(pd.read_csv(complete_path, header=None, dtype=str)[0])

lstm = torch.load('lstm_models/ling_50ms.p')
ffnn = torch.load('smol_from_big.p')

s = nn.Sigmoid()

def find_trps():
    losses_test = list()
    results_dict = dict()
    losses_dict = dict()
    batch_sizes = list()
    trp_dict = dict()
    distance_dict = dict()
    losses_mse, losses_l1 = [], []
    lstm.eval()
    # setup results_dict
    results_lengths = complete_dataset.get_results_lengths()
    for file_name in complete_file_list:
        #        for g_f in ['g','f']:
        for g_f in ['g','f']:
            # create new arrays for the results
            results_dict[file_name + '/' + g_f] = np.zeros([results_lengths[file_name], prediction_length]) # results length -- for each prediction
            trp_dict[file_name] = np.zeros([results_lengths[file_name], 1])
            distance_dict[file_name] = np.zeros([results_lengths[file_name], 1])

    for batch_indx, batch in enumerate(complete_dataloader):
        # print(batch_indx)

        model_input = []

        for b_i, bat in enumerate(batch):
            if len(bat) == 0:
                model_input.append(bat)
            elif (b_i == 1) or (b_i == 3):
                model_input.append(torch.squeeze(bat, 0).transpose(0, 2).transpose(1, 2).numpy())
            elif (b_i == 0) or (b_i == 2):
                model_input.append(Variable(torch.squeeze(bat, 0).type(dtype)).transpose(0, 2).transpose(1, 2))

        y_test = Variable(torch.squeeze(batch[4].type(dtype), 0))

        info_test = batch[-1]
        batch_length = int(info_test['batch_size'])
        if batch_indx == 0:
            lstm.change_batch_size_reset_states(batch_length)
        else:
            if slow_test:
                lstm.change_batch_size_no_reset(batch_length)
            else:
                lstm.change_batch_size_reset_states(batch_length)
        # print(batch_indx)

        out_test = lstm(model_input)
        out_test = torch.transpose(out_test, 0, 1)

        if complete_dataset.set_type == 'test':
            file_name_list = [info_test['file_names'][i][0] for i in range(len(info_test['file_names']))]
            gf_name_list = [info_test['g_f'][i][0] for i in range(len(info_test['g_f']))]
            time_index_list = [info_test['time_indices'][i][0] for i in range(len(info_test['time_indices']))]
        else:
            file_name_list = info_test['file_names']
            gf_name_list = info_test['g_f']
            time_index_list = info_test['time_indices']

        # Should be able to make other loss calculations faster
        # Too many calls to transpose as well. Should clean up loss pipeline

        for file_name, g_f_indx, time_indices, batch_indx in zip(file_name_list,
                                                                 gf_name_list,
                                                                 time_index_list,
                                                                 range(batch_length)):

            # results_dict[file_name + '/' + g_f_indx][time_indices[0]:time_indices[1]] = out_test[
            #     batch_indx][:,:20].data.cpu().numpy()
            results_dict[file_name + '/' + g_f_indx][time_indices[0]:time_indices[1]] = out_test[
                batch_indx].data.cpu().numpy()
            
    ################ elliott gruzin's addition ############################
            
    for file_name in complete_file_list:
        # print(file_name)
        frame_no = 0
        prev_value_g = None
        prev_value_f = None


        for frame_indx, frame in enumerate(results_dict[file_name +'/g']):

            ############# heuristics approach #################
            
            g_prob = np.sum(results_dict[file_name + '/g'][frame_indx,][:20])/20
            f_prob = np.sum(results_dict[file_name + '/f'][frame_indx,][:20])/20

            distance = (g_prob - f_prob)**2

            trp = 0
            
            if distance < 1.5:
                trp = 1

            trp_dict[file_name][frame_indx] = trp
            distance_dict[file_name][frame_indx] = distance
            
            ########## through ffnn

            # g_speech = results_dict[file_name +'/g'][frame_indx]
            # f_speech = results_dict[file_name +'/f'][frame_indx]
            #
            # speech = torch.from_numpy(np.append(g_speech, f_speech).astype(np.float32)).unsqueeze(0).to(device=torch.device('cuda:0'))
            # trp_prediction = s(ffnn(speech))
            # # print(trp_prediction.data.cpu().numpy()[0][0])
            # if trp_prediction.data.cpu().numpy()[0][0] > 0.65:
            #     trp = 1
            # else:
            #     trp = 0
            # trp_dict[file_name][frame_indx] = trp

    return trp_dict, distance_dict

trps, dstn = find_trps()
with open('trp_dictionaries/distance_trps.pkl', 'wb') as trpfile:
    pickle.dump(trps, trpfile)
# with open('distance_scores/sigmoid_0.2_dist_scores.pkl', 'wb') as dstfile:
#     pickle.dump(dstn, dstfile)
