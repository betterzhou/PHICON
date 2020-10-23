import spacy
import numpy as np
import random
import random as rn
from numpy import dot
from numpy.linalg import norm
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from data_handling_for_heuristic import *
import os

from nltk.parse.stanford import StanfordParser
from configparser import ConfigParser

### parsing parameters.ini
config = ConfigParser()
config.read('parameters.ini')
read_file_path = config.get('file-path', 'read_file_path')
filter_none_entity = config.getboolean('replace', 'filter_none_entity')
window_size_replace = config.getint('replace', 'window_size')
alpha = config.getfloat('replace', 'alpha')
sim_thr = config.getfloat('replace', 'sim_thr')
n_candidates_replace = config.getint('replace', 'n_candidates')
window_size_insert = config.getint('insert', 'window_size')
n_candidates_insert = config.getint('insert', 'n_candidates')

### load resources
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
nlp = spacy.load("en_vectors_web_lg")
nlp_2th = spacy.load("en_core_web_sm")
os.environ['STANFORD_PARSER'] = 'D:\\stanford-parser-full-2016-10-31\\stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'D:\\stanford-parser-full-2016-10-31\\stanford-parser-3.7.0-models.jar'
cons_parser = StanfordParser(model_path = 'edu\\stanford\\nlp\\models\\lexparser\\englishPCFG.ser.gz')

### load data
raw_data, label_data = load_conll2003(read_file_path)


class Test():
    def __init__(self):
        print((random.getstate()))

        
class Insert():

    def __init__(self):
        if filter_none_entity == True:
            self.sent_raw, self.sent_label = filtering_none_entity(raw_data, label_data)
        else:
            self.sent_raw = raw_data[:]
            self.sent_label = label_data[:] 
        self.friend_list = load('resources/for_insert/friend_list')
        self.voca = load('resources/for_insert/voca')
        print('> [Insert]: class is created')
    
    def similarity(self, me, other):
        me = nlp.vocab[me]
        other = nlp.vocab[other]
        if me.vector_norm == 0 or other.vector_norm == 0:
            return 0.0
        return np.dot(me.vector, other.vector) / (me.vector_norm * other.vector_norm)    
    
    def max_friend(self, friend_list, token):
        sim_score = [0] * len(friend_list)
        for i, each in enumerate(friend_list):
            sim_score[i] = self.similarity(token, friend_list[i])
        sim_score.sort(key=lambda x: x, reverse=True)
        sim_score = sim_score[:n_candidates_insert]
        sim_score = list(filter(lambda x: x!= 0.0, sim_score))
        if len(sim_score)==0:
            return 'none'
        #print(sim_score)
        #print('\n')
        #print(np.array(sim_score).max())
        choiced_score = random.choice(sim_score)
        #print(choiced_score)
        #print(np.array(sim_score).max(), '--', choiced_score)
        max1_index = np.where(sim_score == np.array(sim_score).max())[0][0]
        max_index = np.where(sim_score == choiced_score)[0][0]
        #print(max1_index, max_index)
        #print('\n')
        return friend_list[max_index]
          
    def insert_module(self, token, pos):
        #print('------>',token, pos)
        try:voca_index = self.voca.index((token, pos))
        except(ValueError):
            #print('*********>', 'none')
            return 'none'
        if len(self.friend_list[voca_index])==0:
            #print('*********>', 'none')
            return 'none'
        else:
            new_token = self.max_friend(self.friend_list[voca_index], token)
            #print('*********>',new_token)
            return new_token        

    def window_filtering(self, sent_label, window_size=3):
        list_sent_label = sent_label.split()
        new_list = [0] * len(list_sent_label)
        ### 순방향
        for i, _ in enumerate(list_sent_label):
            temp = list_sent_label[i:i+1+window_size]
            #print(temp)
            tkn = True
            # for each in temp:
            #     if each != 'O':
            #         tkn = True
            if tkn == True:
                new_list[i] = 1
        ### 역방향
        for i  in range(len(list_sent_label)-1, -1, -1):
            idx = i-window_size
            if idx < 0:
                idx = 0
            temp = list_sent_label[idx:i]
            tkn = True
            # for each in temp:
            #     if each != 'O':
            #         tkn = True
            if tkn == True:
                new_list[i] = 1      
        return new_list  

    def do(self, data):
        sent_raw = data[0]
        sent_label = data[1]
        print('>>> [Insert Start!]: (number of input sent = {})'.format(len(sent_raw)))
        insert_sent_cnt = 0
        new_sent_list = []
        new_label_list = [] 
        for i, _ in enumerate(sent_raw):
            #list_sent_raw = sent_raw[i].split()
            str_sent_raw = preprocessing_for_spacy(sent_raw[i])
            list_sent_nlp = nlp_2th(str_sent_raw)  # add new Spacy model to get word type
            list_sent_raw = str_sent_raw.split()
            list_sent_label = sent_label[i].split()
            #print(str_sent_raw)
            ### 조건에 맞는 token들을 불러온다
            pre_token_pos = 'none'
            insert_on = False
            add_idx = 0
            ### filtering with window size based on entities
            filtered_list = self.window_filtering(sent_label[i], window_size_insert)
            #print(sent_raw[i])
            #print(sent_label[i])
            if len(list_sent_nlp) != len(list_sent_label):
                #print('[ERROR]: len(list_sent_nlp) != len(list_sent_label)')
                continue
            for i, token in enumerate(list_sent_nlp):
                new_token = 'none' # 계속 'none'으로 남아있다면 추가하지 않을 것임.
                # 엔티티는 건드리지 말자.
                if list_sent_label[i+add_idx] != 'O':
                    continue
                if filtered_list[i+add_idx] == 0: # filtering with window size based on entities
                    continue
                #print('>>>', token, token.pos_, pre_token_pos)
                #################
                ### INSERT_MODULE  
                str_token = str(token).lower()
                if token.pos_ == 'NOUN' or token.pos_ == 'PROPN': # 오로지 하나의 명사일경우...
                    if i==0:
                        str_token = lemmatiser.lemmatize(str_token, 'n')
                        new_token = self.insert_module(str_token, 'noun')
                    else:
                        if pre_token_pos != 'ADJ':
                            if pre_token_pos != 'NOUN':
                                if pre_token_pos != 'PROPN':
                                    str_token = lemmatiser.lemmatize(str_token, 'n')
                                    new_token = self.insert_module(str_token, 'noun')
                elif token.pos_ == 'VERB': 
                    str_token = lemmatiser.lemmatize(str_token, 'v')
                    new_token = self.insert_module(str_token, 'verb')
                elif token.pos_ == 'ADJ':
                    new_token = self.insert_module(str_token, 'adj')
                
                ################
                ### REAL INSERT
                pre_token_pos = token.pos_
                if new_token == 'none': # insert된 것이 하나도 없으면 pass.
                    continue # pass
                else:
                    insert_on = True # insert 한 번 이상 되었음.
                    list_sent_raw.insert(i+add_idx, new_token)
                    list_sent_label.insert(i+add_idx, 'O')
                    filtered_list.insert(i+add_idx, -1)
                    add_idx += 1
                #print('\n')
            #print(' '.join(list_sent_raw))
            #print(' '.join(list_sent_label))
            #print('\n')
            ### Append new sent list
            if insert_on == True:
                insert_sent_cnt += 1
                new_sent_list.append(' '.join(list_sent_raw))
                new_label_list.append(' '.join(list_sent_label))
            #break
        print('>>> [Insert Done!]: (number of output (new) sent = {})'.format(insert_sent_cnt))            
        return new_sent_list, new_label_list, insert_sent_cnt
    
    
class Replace():

    def __init__(self):
        print('> [Replace]: class is created')
        
    def wordnet(self, word, pos):
        if type(word) != 'str':
            word = str(word)   
        synsets = wn.synsets(word, pos)
        list_synsets = []
        for i, syn in enumerate(synsets):
            #print('%d. %s' % (i, syn.name()))
            li = (syn.lemma_names())
            list_synsets += (li)
            for hyper_syn in syn.hypernyms():
                list_synsets += (hyper_syn.lemma_names())
            for hypo_syn in syn.hyponyms():
                list_synsets += (hypo_syn.lemma_names())
            for holo_syn in syn.part_holonyms():
                list_synsets += (holo_syn.lemma_names())
            for mero_syn in syn.part_meronyms():
                list_synsets += (mero_syn.lemma_names())

        # 똑같은 stem 제거
        source_stem = stemmer.stem(word)
        source_lemma = lemmatiser.lemmatize(word, pos)
        for i, token in enumerate(list_synsets):
            #print(source_stem, stemmer.stem(str(token)), source_lemma, lemmatiser.lemmatize(str(token), pos))
            if source_stem == stemmer.stem(token):
                list_synsets[i] = 'to-be-deleted'
            if source_lemma == lemmatiser.lemmatize(token, pos):
                list_synsets[i] = 'to-be-deleted'
            if len(token.split('_')) != 1: # 'bill_of_exchange'와 같이 명사구 또는 복합명사일 경우... 삭제...
                list_synsets[i] = 'to-be-deleted'
        list_synsets = [x for x in list_synsets if x != 'to-be-deleted'] # 한번에 삭제      
        return list_synsets   
    
 
    def word2vec(self, word, pos, synonym_select):
        thr = sim_thr
        thr = thr / 2.0
        if type(word) != 'str':
            word = str(word)
        n_candidates = n_candidates_replace
        nlp_word = nlp.vocab[word]
        
        ### similarity가 비슷한 단어들 추출
        queries = [w for w in nlp_word.vocab if w.is_lower == nlp_word.is_lower and w.prob >= -11]
        for w in queries:
            if not w.vector_norm:
                queries.remove(w)
        by_similarity = sorted(queries, key=lambda w: nlp_word.similarity(w), reverse=True)
        #print(len([w.lower_ for w in by_similarity[:]]))
        #print(([w.lower_ for w in by_similarity[:]]))
        cand_word_list = [w.lower_ for w in by_similarity[:n_candidates]]  # some candidate words
        #print(cand_word_list)
        
        ### 후보군들 중에서 어떤 것을 교체어로 선정할 것인가?
        # stemming과 lemmatization을 모두 사용하자.
        source_stem = stemmer.stem(word)
        source_lemma = lemmatiser.lemmatize(word, pos)
        
        for i, token in enumerate(cand_word_list):
            #print(source_stem, stemmer.stem(str(token)), source_lemma, lemmatiser.lemmatize(str(token), pos))
            if source_stem == stemmer.stem(token):
                cand_word_list[i] = 'to-be-deleted'
            if source_lemma == lemmatiser.lemmatize(token, pos):
                cand_word_list[i] = 'to-be-deleted'
        cand_word_list = [x for x in cand_word_list if x != 'to-be-deleted'] # 한번에 삭제

        # print('cand_word_list:', cand_word_list)
        print('cand_word_list has been generated, please wait for a while !')
        if len(cand_word_list) == 0:
            return False
        if nlp_word.similarity(nlp.vocab[cand_word_list[0]]) > thr: # 70%로 비슷하면 
            return cand_word_list[synonym_select] # 최상위 1개 선택
            #return random.choice(cand_word_list) # 랜덤으로 선택
        else:
            list_wordnet = self.wordnet(word, pos)
            cand_word_list_filtered_from_wornet = [v for v in cand_word_list if v in list_wordnet]
            if len(cand_word_list_filtered_from_wornet)==0:
                if len(list_wordnet)==0:
                    return False
                else:
                    return list_wordnet[0]
            else:
                #print(cand_word_list_filtered_from_wornet)
                #print('\n')
                #return random.choice(cand_word_list_filtered_from_wornet)
                return cand_word_list_filtered_from_wornet[0]    

    def replace_algorithm(self, list_sent_raw, list_rep, pos, synonym_select):
        bit = False
        new_list_sent_raw = list_sent_raw[:]
        for token, _, idx in list_rep:
            result = self.word2vec(token, pos, synonym_select)
            if result != False: # 실패하지 않은 경우에만...
                bit = True
                new_list_sent_raw[idx] = result
        return new_list_sent_raw, bit

    def window_filtering(self, sent_label, window_size=3):
        list_sent_label = sent_label.split()
        new_list = [0] * len(list_sent_label)
        ### 순방향
        for i, _ in enumerate(list_sent_label):
            temp = list_sent_label[i:i+1+window_size]
            #print(temp)
            tkn = True
            # for each in temp:
            #     if each != 'O':
            #         tkn = True
            if tkn == True:
                new_list[i] = 1
        ### 역방향
        for i  in range(len(list_sent_label)-1, -1, -1):
            idx = i-window_size
            if idx < 0:
                idx = 0
            temp = list_sent_label[idx:i]
            tkn = True
            # for each in temp:
            #     if each != 'O':
            #         tkn = True
            if tkn == True:
                new_list[i] = 1      
        return new_list    


    def generation_via_replace(self, sent_raw, sent_label, alpha, synonym_select):

        #############
        """ 전처리 """
        #############
        # 동사, 명사는 context 정보의 핵심이다 모든 언어가 공통적으로 가지고 있는 특징.  
        #print(sent_raw)
        list_sent_raw = sent_raw.split()
        sent_raw = preprocessing_for_spacy(sent_raw)
        sent_nlp = nlp_2th(sent_raw)  # add new type of Spacy model, en_core_web_sm
        noun_sorted, verb_sorted, adj_sorted, adv_sorted = [], [], [], []
        
        """
        ### assert 에러 났을 때, 원인 찾기 위해 print 실시
        if not len(sent_nlp) == len(sent_label.split()):
            print(len(sent_raw.split()), len(sent_nlp), len(sent_label.split()))
            print(sent_raw)
            print(sent_nlp)
            for i, token in enumerate(sent_nlp):
                print(token, sent_raw.split()[i], sent_label.split()[i])
        assert(len(sent_nlp) == len(sent_label.split()))
        """
        ### filtering with window size based on entities
        filtered_list = self.window_filtering(sent_label, window_size_replace)

        
        if len(sent_nlp) != len(sent_label.split()):
            #print('[ERROR]: len(list_sent_nlp) != len(list_sent_label)')
            return False    
        
        ### 조건에 맞는 token들을 불러온다 
        for i, token in enumerate(sent_nlp):
            ### 제외되는 token들..
            if not sent_label.split()[i] == 'O': # 엔티티가 아닌 token은 제외
                continue
            is_uppercase_letter = True in map(lambda l: l.isupper(), str(token))
            if i != 0 and is_uppercase_letter == True: # 첫 번째 글자가 아니고 대문자가 하나라도 있는 token은 제외 (ex. Thursday와 같은 시간고유명사. 크게 의미적이지 않음.)
                continue
            if lemmatiser.lemmatize(str(token), 'v') == 'be': # be동사 lemma를 가지는 token은 제외
                continue
            if lemmatiser.lemmatize(str(token), 'v') == 'have': # have 동사 lemma를 가지는 token은 제외, (have 동사의 의미가 광범위하게 쓰이기도 하고, have가 분사형으로 사용되기도 하기 때문이다)
                continue
            if filtered_list[i] == 0: # filtering with window size based on entities
                continue
                
            ### 동사, 명사, 형용사, 부사 추출
            if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
                pair = (sent_nlp[i], len(list(token.subtree)), i) 
                noun_sorted.append(pair) # 명사
                   
            elif token.pos_ == 'VERB':
                pair = (sent_nlp[i], len(list(token.subtree)), i)
                verb_sorted.append(pair) # 동사       
            
            elif token.pos_ == 'ADJ':
                pair = (sent_nlp[i], len(list(token.subtree)), i)
                adj_sorted.append(pair) # 형용사         

            elif token.pos_ == 'ADV':
                pair = (sent_nlp[i], len(list(token.subtree)), i)
                adv_sorted.append(pair) # 부사   
        
        ### dependency 크기로 정렬
        noun_sorted.sort(key=lambda x: x[1], reverse=True)
        verb_sorted.sort(key=lambda x: x[1], reverse=True)
        adj_sorted.sort(key=lambda x: x[1], reverse=True)
        adv_sorted.sort(key=lambda x: x[1], reverse=True)
        
    #     print((noun_sorted))
    #     print((verb_sorted))
        if len(noun_sorted)==0 and len(verb_sorted)==0 and len(adj_sorted)==0 and len(adv_sorted)==0:
            return False # 교체 후보 token이 아예없다면, 새로운 문장을 만들 수 없다.

        ### thr 정하기
        noun_thr = int(len(noun_sorted) * alpha)
        verb_thr = int(len(verb_sorted) * alpha)
        adj_thr = int(len(adj_sorted) * alpha)
        adv_thr = int(len(adv_sorted) * alpha)
        
        if noun_thr == 0:
            noun_thr = 1 # 최소값은 1로 유지
        if verb_thr == 0:
            verb_thr = 1 # 최소값은 1로 유지
        if adj_thr == 0:
            adj_thr = 1 # 최소값은 1로 유지
        if adv_thr == 0:
            adv_thr = 1 # 최소값은 1로 유지        
        
        ### token list to be replaced
        noun_rep = noun_sorted[:noun_thr] 
        verb_rep = verb_sorted[:verb_thr] 
        adj_rep = adj_sorted[:adj_thr]
        adv_rep = adv_sorted[:adv_thr]
        
        #print(noun_rep)
        #print(verb_rep)
        #print(list_sent_raw)
        
        if len(noun_sorted) != 0: # 각자 0이 될 수도 있으니...
            list_sent_raw, bit_n = self.replace_algorithm(list_sent_raw, noun_rep, 'n', synonym_select)
            #print(list_sent_raw)
            #replaced_noun = (noun_sorted[0][0], noun_sorted[0][2]) # (token, index) pair 
            # 명사
            #print('<<< ', replaced_noun[0], '  >>>')
            #target_noun = replace_with_word2vec(replaced_noun[0], 'n')
            #print(replace_with_sense2vec(replaced_noun[0], 'NOUN'))
            #print(replace_with_wordnet(replaced_noun[0], 'n'))
            #print('replaced_noun: ', replaced_noun, ', target_noun: ', target_noun)
            #list_sent_raw[replaced_noun[1]] = target_noun # 교체
        else:
            bit_n = False
        if len(verb_sorted) != 0: # 각자 0이 될 수도 있으니...
            list_sent_raw, bit_v = self.replace_algorithm(list_sent_raw, verb_rep, 'v', synonym_select)
            #print(list_sent_raw)
            #replaced_verb = (verb_sorted[0][0], verb_sorted[0][2])
            # 동사
            #print('<<< ', replaced_verb[0], '  >>>')
            #target_verb = replace_with_word2vec(replaced_verb[0], 'v')
            #print(replace_with_sense2vec(replaced_verb[0], 'VERB'))
            #print(replace_with_wordnet(replaced_verb[0], 'v'))   
            #print('replaced_verb: ', replaced_verb, ', target_verb: ', target_verb)
            #list_sent_raw[replaced_verb[1]] = target_verb # 교체
        else:
            bit_v = False
        if len(adj_sorted) != 0: # 각자 0이 될 수도 있으니...
            list_sent_raw, bit_a = self.replace_algorithm(list_sent_raw, adj_rep, 'a', synonym_select)
        else:
            bit_a = False    
        if len(adv_sorted) != 0: # 각자 0이 될 수도 있으니...
            list_sent_raw, bit_r = self.replace_algorithm(list_sent_raw, adv_rep, 'r', synonym_select)
        else:
            bit_r = False 
        if bit_n==False and bit_v==False and bit_a==False and bit_a==False:
            return False
        else:    
            #print(' '.join(list_sent_raw))
            return ' '.join(list_sent_raw)    
    
    def do(self, data, synonym_select):
        sent_raw = data[0]
        sent_label = data[1]
        print('>>> [Replace Start!]: (number of input sent = {})'.format(len(sent_raw)))
    
        new_sent_raw_list = []
        new_sent_label_list = [] 
        for i, row in enumerate(sent_raw):
            alpha = 0.9
            result = self.generation_via_replace(sent_raw[i], sent_label[i], alpha, synonym_select)
            if result == False: # 교체어가 없었는 경우
                continue
            else:
                new_sent_raw_list.append(result)
                new_sent_label_list.append(sent_label[i]) # 라벨은 동일하다.    
        print('>>> [Replace Done!] (number of output (new) sent = {}'.format(len(new_sent_raw_list)))
        return new_sent_raw_list, new_sent_label_list, len(new_sent_raw_list)
        

        
        
