import pickle
import pprint
import matplotlib.pyplot as plt
import numpy as np
import wave
import math
import re
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

jar = open('trp_dictionaries/distance_trps.pkl','rb')
trp_dict = pickle.load(jar)

# print(trp_dict)
# exit()
def round_nearest(x, a):
    return round(x / a) * a


available = 0
correct = 0
in_range = 0
total = 0

what_correct = {}
what_count = {}

def extract_eval(file_name):
    # print(file_name)
    global correct_distances
    global incorrect_distances
    global available
    global what_correct
    global correct
    global in_range
    global total
    trps = trp_dict[file_name]
    file_length = len(trps)/20
    # print(len(trps))
    # print(len(distances))
    trp_times = []
    for i, trp in enumerate(trps):
        if trp == 1:
            trp_times.append(i/20)


    da_dict = {}

    def get_dial_act(file):
        with open(file,'r') as input:
            for line in input:

                match = re.search(r'End: (\d+\.\d{3})\d+\s+DA: (\w+)', line)
                da_time = match.group(1)
                da_time = "%0.2f"%round_nearest(float(da_time),0.05)
                da_type = match.group(2)
                if da_type != 'abandon':
                    da_dict[int(float(da_time)*20)] = da_type


    get_dial_act('dial_acts_transcriptions/{}.f.txt'.format(file_name))
    get_dial_act('dial_acts_transcriptions/{}.g.txt'.format(file_name))

    gs_trps = [0]*len(trps)
    for trp in da_dict.keys():
        if trp < len(trps):
            gs_trps[trp] = 1
    # print(gs_trps)
    # exit()

    pairs = sorted(list(da_dict.items()))

    # pprint.pprint(sorted(pairs))
    #
    # ### recall calc
    # print(da_dict.keys())
    # print(trp_times)

    ### refine trps to take earliest!
    # prev_time = 0
    # trp_time_temp = []
    # for t in trp_times:
    #     if t-prev_time <= 0.1:
    #         prev_time = t
    #     else:
    #         prev_time = t
    #         trp_time_temp.append(t)
    # # print(trp_time_temp)
    # trp_times = trp_time_temp
    global_mismatch = 0
    wd_mismatch = 0

    gs_bin = []
    c_bin = []

    for i in range(len(trps)-13):
        gs_strip = gs_trps[i:i+15]
        candidate_strip = trps[i:i+15]
        gs_val = sum(gs_strip)
        c_val = sum(candidate_strip)
        if gs_val != c_val:
            wd_mismatch += 1
        if 1 in gs_strip and 1 not in candidate_strip:
            global_mismatch += 1
        if 1 not in gs_strip and 1 in candidate_strip:
            global_mismatch += 1
        if 1 in gs_strip:
            gs_bin.append(1)
        if 1 not in gs_strip:
            gs_bin.append(0)
        if 1 in candidate_strip:
            c_bin.append(1)
        if 1 not in candidate_strip:
            c_bin.append(0)
    return global_mismatch, wd_mismatch, gs_bin, c_bin, len(trps)-13


# for da_time, da_act in pairs:
#     print('Time: {0}\t DA: {1}\t Captured: {2}'.format(da_time, da_dict[da_time], when_correct[da_time]))

dev_set = open('data/splits/trp_prediction_sets/trp_test.txt','r')

mm = 0
no_frames = 0
wd = 0
candidate_binaries = []
gold_standard_binaries = []


for line in dev_set:
    file_mm, file_wd, file_gs_bin, file_c_bin, file_frames = extract_eval(line[:-1])
    mm += file_mm
    wd += file_wd
    no_frames += file_frames
    candidate_binaries.extend(file_c_bin)
    gold_standard_binaries.extend(file_gs_bin)

p0gs = sum(gold_standard_binaries)/no_frames
p0c = sum(candidate_binaries)/no_frames

c = p0gs*p0c + (1-p0gs)*(1-p0c)
k_min_kappa = (1 - (mm/no_frames) - c)/(1 - c)

ref = np.array(gold_standard_binaries)
hyp = np.array(candidate_binaries)

k_precision = precision_score(ref,hyp)
k_recall = recall_score(ref, hyp)

print('\n\n\n ************************** \n\n\n')
print('Pk score: ' + str(mm/no_frames))
print('WD score: ' + str(wd/no_frames))
print('k-kappa: ' + str(k_min_kappa))
print('k-precision: '+str(k_precision))
print('k_recall: ' + str(k_recall))
print('\n\n\n ************************** \n\n\n')


# print('General distance ', general_distance)
# print('Expected/Correct Distances: ', str(sum(correct_distances)/len(correct_distances)))
# print('Expected/Incorrect Distances: ', str(sum(incorrect_distances)/len(incorrect_distances)))
