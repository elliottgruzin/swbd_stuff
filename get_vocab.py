import xml.etree.ElementTree
import os
import numpy as np
import pandas as pd
import time as t
import pickle
import sys
import json
import nltk
nltk.download('punkt')


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
t_1 = t.time()


path_to_features='../switchboard_version/data/signals/gemaps_features_processed_50ms/znormalized/'
path_to_annotations='./nxt/xml/terminals/'
path_to_extracted_annotations='./voice_activity/'
files_feature_list = os.listdir(path_to_features)
files_annotation_list = list()
files_output_list = list()

########## note that A = g, B = f ###########


for file in files_feature_list:
    base_name = os.path.basename(file)
    num = base_name.split('.')[0][3:]
    if base_name.split('.')[1] == 'g':
        speaker = 'A'
    elif base_name.split('.')[1] == 'f':
        speaker = 'B'
    files_annotation_list.append('nxt/xml/terminals/sw{}.{}.terminals.xml'.format(num,speaker))

#%% Get vocabulary
no_change, disfluency_count,multi_word_count = 0,0,0
words_from_annotations = []
for i in range(0,len(files_feature_list)):
    # sys.stdout.flush()
    print('percent done vocab build:'+str(i/len(files_feature_list))[0:4])
    e = xml.etree.ElementTree.parse(files_annotation_list[i]).getroot()
    for atype in e.findall('word'):
        target_word = atype.get('orth')
        target_word = target_word.strip()
        if '--' in target_word:
            target_word ='--disfluency_token--'
            words_from_annotations.append(target_word)
            disfluency_count += 1
        else:
            target_words = nltk.word_tokenize(target_word)
            words_from_annotations.extend( target_words)

vocab = set(words_from_annotations)
word_to_ix = {word: i+1 for i, word in enumerate(vocab)} # +1 is because 0 represents no change
ix_to_word = {word_to_ix[wrd]: wrd for wrd in word_to_ix.keys()}
pickle.dump(word_to_ix,open('word_to_ix.p','wb'))
pickle.dump(ix_to_word,open('ix_to_word.p','wb'))
json.dump(word_to_ix,open('word_to_ix.json','w'),indent=4)

print('disfluency count: '+str(disfluency_count))
print('total_time: '+str(t.time()-t_1))
