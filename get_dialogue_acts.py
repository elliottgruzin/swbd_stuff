import xml.etree.ElementTree
import os
import numpy as np
import time as t


# for file in files_feature_list:

da_files = os.listdir('nxt/xml/dialAct')

terminal_files = []
files_output_list = []

nite = '{http://nite.sourceforge.net/}'

for file in da_files:
    num = file.split('.')[0][2:]
    # print(file)
    if file.split('.')[1] == 'A':
        speaker = 'g'
    elif file.split('.')[1] == 'B':
        speaker = 'f'
    terminal_files.append('nxt/xml/terminals/{}.terminals.xml'.format(file[:8]))
    files_output_list.append('dial_acts_transcriptions/sw0{}.{}.txt'.format(num,speaker))
skip_set = set()
for i in range(len(da_files)):
    with open(files_output_list[i], 'w') as output:
        dialacts = []
        da_file = xml.etree.ElementTree.parse('nxt/xml/dialAct/'+da_files[i]).getroot()
        term_file = xml.etree.ElementTree.parse(terminal_files[i]).getroot()

        for da in da_file.findall('da'):
            act_type = da.get('niteType')
            transcript = []
            words = da.findall(nite+'child')
            try:
                begin_id = words[0].attrib['href'].split('#')[1]
                first_word = term_file.findall('./word[@{}id="{}"]'.format(nite,begin_id[3:-1]))[0]
                begin_time = first_word.get(nite+'start')
                end_id = words[-1].attrib['href'].split('#')[1]
                last_word = term_file.findall('./word[@{}id="{}"]'.format(nite,end_id[3:-1]))[0]
                end_time = last_word.get(nite+'end')
            except IndexError:
                print('Cannot align times -- skip file ' + files_output_list[i])
                skip_set.add(files_output_list[i][:-6])
                continue
            for word in words:
                id = word.attrib['href'].split('#')[1]
                try:
                    word_element = term_file.findall('./word[@{}id="{}"]'.format(nite,id[3:-1]))[0]
                except IndexError:
                    continue
                transcript.append(word_element.get('orth'))
            output.write('Begin: {}\t End: {}\t DA: {}\t Transcript: {}.\n'.format(
            begin_time,end_time, act_type ,' '.join(transcript)
            ))
print(skip_set)
print(len(skip_set))
