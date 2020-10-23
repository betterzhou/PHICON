###############################################
### Parameter Setting
ref_read_path = 'data/conll2003'
ref_write_path = 'gen_data/copy/subset'

###############################################

from data_handling_for_heuristic import *
import os, sys
import random

arg_str = ' '.join(sys.argv[1:])

percent_subset = arg_str


### Train Dataset
# Read
read_path = ref_read_path + '/train.txt'
raw_data, label_data = load_conll2003(read_path)

temp_pair = []
for i, _ in enumerate(raw_data):
    each = (raw_data[i], label_data[i])
    temp_pair.append(each)

random.shuffle(temp_pair)
output = temp_pair[:int(len(temp_pair)*float(percent_subset))]
print('before:', len(temp_pair), '--->', ' after:', len(output))

# Write
path_write = ref_write_path+'/train_'+str(percent_subset)+'_'+str(len(output))+'.txt'
with open(path_write, 'w', encoding='UTF-8') as txt:    
    for i, _ in enumerate(output):
        splited_sent = output[i][0].split()
        splited_label = output[i][1].split()
        for j, token in enumerate(splited_sent):
            txt.write(splited_sent[j]+' '+'NNP'+' '+'B-NP'+' '+splited_label[j])
            txt.write('\n')
        txt.write('\n')

  
### Valid Dataset
# Read
read_path = ref_read_path +'/valid.txt'
raw_data, label_data = load_conll2003(read_path)

temp_pair = []
for i, _ in enumerate(raw_data):
    each = (raw_data[i], label_data[i])
    temp_pair.append(each)

random.shuffle(temp_pair)
output = temp_pair[:int(len(temp_pair)*float(percent_subset))]
print('before:', len(temp_pair), '--->', ' after:', len(output))

# Write
path_write = ref_write_path+'/valid_'+str(percent_subset)+'_'+str(len(output))+'.txt'
with open(path_write, 'w', encoding='UTF-8') as txt:    
    for i, _ in enumerate(output):
        splited_sent = output[i][0].split()
        splited_label = output[i][1].split()
        for j, token in enumerate(splited_sent):
            txt.write(splited_sent[j]+' '+'NNP'+' '+'B-NP'+' '+splited_label[j])
            txt.write('\n')
        txt.write('\n')  


### Test Dataset
# Read
read_path = ref_read_path +'/test.txt'
raw_data, label_data = load_conll2003(read_path)

temp_pair = []
for i, _ in enumerate(raw_data):
    each = (raw_data[i], label_data[i])
    temp_pair.append(each)

random.shuffle(temp_pair)
output = temp_pair[:int(len(temp_pair)*float(percent_subset))]
print('before:', len(temp_pair), '--->', ' after:', len(output))

# Write
path_write = ref_write_path+'/test_'+str(percent_subset)+'_'+str(len(output))+'.txt'
with open(path_write, 'w', encoding='UTF-8') as txt:    
    for i, _ in enumerate(output):
        splited_sent = output[i][0].split()
        splited_label = output[i][1].split()
        for j, token in enumerate(splited_sent):
            txt.write(splited_sent[j]+' '+'NNP'+' '+'B-NP'+' '+splited_label[j])
            txt.write('\n')
        txt.write('\n')        