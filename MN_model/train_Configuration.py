# -*- coding: utf-8 -*-
import sys

# copy and paste from Log file in the data folder
# ALLPHONEMES = ['a', 'aa', 'ae', 'ax', 'b', 'bh', 'blank', 'c', 'ch', 'd', 'dh', 'dx', 'dxh', 'dxhq', 'dxq', 'ee', 'ei', 'g', 'gh', 'h', 'i', 'ii', 'j', 'jh', 'k', 'kh', 'l', 'lx', 'm', 'mq', 'n', 'nj', 'nx', 'o', 'ou', 'p', 'ph', 'q', 'r', 'rq', 's', 'sh', 'sil', 'sx', 't', 'th', 'tx', 'txh', 'u', 'uu', 'w', 'y', 'z']

# index_dict = {'blank': 0, 'sil': 1, 'a': 2, 'aa': 3, 'ae': 4, 'ax': 5, 'b': 6, 'bh': 7, 'c': 8, 'ch': 9, 'd': 10, 'dh': 11, 'dx': 12, 'dxh': 13, 'dxhq': 14, 'dxq': 15, 'ee': 16, 'ei': 17, 'g': 18, 'gh': 19, 'h': 20, 'i': 21, 'ii': 22, 'j': 23, 'jh': 24, 'k': 25, 'kh': 26, 'l': 27, 'lx': 28, 'm': 29, 'mq': 30, 'n': 31, 'nj': 32, 'nx': 33, 'o': 34, 'ou': 35, 'p': 36, 'ph': 37, 'q': 38, 'r': 39, 'rq': 40, 's': 41, 'sh': 42, 'sx': 43, 't': 44, 'th': 45, 'tx': 46, 'txh': 47, 'u': 48, 'uu': 49, 'w': 50, 'y': 51, 'z': 52}

# label_dict = {0: 'blank', 1: 'sil', 2: 'a', 3: 'aa', 4: 'ae', 5: 'ax', 6: 'b', 7: 'bh', 8: 'c', 9: 'ch', 10: 'd', 11: 'dh', 12: 'dx', 13: 'dxh', 14: 'dxhq', 15: 'dxq', 16: 'ee', 17: 'ei', 18: 'g', 19: 'gh', 20: 'h', 21: 'i', 22: 'ii', 23: 'j', 24: 'jh', 25: 'k', 26: 'kh', 27: 'l', 28: 'lx', 29: 'm', 30: 'mq', 31: 'n', 32: 'nj', 33: 'nx', 34: 'o', 35: 'ou', 36: 'p', 37: 'ph', 38: 'q', 39: 'r', 40: 'rq', 41: 's', 42: 'sh', 43: 'sx', 44: 't', 45: 'th', 46: 'tx', 47: 'txh', 48: 'u', 49: 'uu', 50: 'w', 51: 'y', 52: 'z'}

# ctc_labels = ['_', 'sil', 'a', 'aa', 'ae', 'ax', 'b', 'bh', 'c', 'ch', 'd', 'dh', 'dx', 'dxh', 'dxhq', 'dxq', 'ee', 'ei', 'g', 'gh', 'h', 'i', 'ii', 'j', 'jh', 'k', 'kh', 'l', 'lx', 'm', 'mq', 'n', 'nj', 'nx', 'o', 'ou', 'p', 'ph', 'q', 'r', 'rq', 's', 'sh', 'sx', 't', 'th', 'tx', 'txh', 'u', 'uu', 'w', 'y', 'z']


# Training
seed=25
use_cuda= True

datapath = sys.argv[1] +"/"

# if balanced:
#     datapath = "./MN_balancedData/"
# else:
#     datapath = "./MN_unbalancedData/"
    
SOURCE_LANGUAGES = ["Hindi","Gujarati","Marathi"]
for source in SOURCE_LANGUAGES:
    datapath+=source[0:3]
datapath+="_"

trainSupportSet = datapath+"bilstm_trainSS_dim39.pkl"
trainQuerySet = datapath+'train_xz.pkl'
devQuerySet= datapath+'dev_xz.pkl'
testQuerySet = datapath+'test_xz.pkl'

#P-way Q-shot
Q=20

prev_model_epochs=0#currnet model epoch number if any
prev_model_path=""#current model generated path if any

epochs=20 #Number of epochs need to be run now
batch_size = 10

model_store_path="./MN_model/Multi_MNCTC_"+str(Q)+"shots_episodic_batch" + str(batch_size)
log_file_path=model_store_path+"/trainlog"