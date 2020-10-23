"""
This code is adapted from the gritmind/train-data-augmentation-for-ner github repo with several modifications.

Notice:
For SR (Synonym Replacement), users need to select a different synonym candidate, when the augmentation factor > 1.
By default, it selects the most similar synonym.
For example, when the augmentation factor = 2, set synonym_candidate_select = 0,1, respectively.
"""

from rule_modules import Replace, Insert
from data_handling_for_heuristic import *

print('\n********** Prepare Dataset **********')
source_data_set = '2006_5_category'
read_file_path = 'data/' + source_data_set + '/train.txt'
raw_data, label_data = load_conll2003(read_file_path)
ori_num = len(raw_data)
tot_new_num = 0

sent_raw1, sent_label1 = filtering_noENT_sentFORM(raw_data, label_data)
print('START: the number of original data: {}'.format(ori_num))

print('\n********** Data Augmentation **********')
total_new_data = []
total_new_label = []

synonym_candidate_select = 0
replace = Replace()
data_rep, label_rep, n_rep = replace.do([sent_raw1, sent_label1], synonym_candidate_select)

insert = Insert()
data_ins, label_ins, n_ins = insert.do([sent_raw1, sent_label1])

print('\n********** Augmented Data Merge & Store **********')
total_new_data = total_new_data + data_rep + data_ins
total_new_label = total_new_label + label_rep + label_ins
tot_new_num = tot_new_num + n_rep + n_ins


path_write = 'gen_data/2020_i2b2_augmentation/' \
              + '2006_5_category_Context_Aug_' + str(tot_new_num) + '.txt'
store_new_sent(path_write, total_new_data, total_new_label)


