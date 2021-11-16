#!/usr/bin/env python
# coding: utf-8

# MN-CTC inference on target language

import inference_Configuration as config
from episodic_dataloader.ctc_batch_loader import QueryDataSet, batch_stackup, pad_batch
from episodic_dataloader.ctc_support_set_loader import SupportDataSet, collate_wrapper, stackup

import numpy as np
import os
import pickle
import random
import sys
from datetime import datetime
from jiwer import wer
from ctcdecode import CTCBeamDecoder
import torch
import torchvision
from torch import nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm
import logging

EPSILON = 1e-8
seed=config.seed
torch.manual_seed(seed)
np.random.seed(seed)

# lstm embedding - encoder_f (Batch utterances)
class few_shot_lstm_encoder(nn.Module):
    def __init__(self, lstm_inp_dim):
        super(few_shot_lstm_encoder, self).__init__()
        
        self.count = 0
        
        # LSTM parameters
        self.lstm_inp_dim = lstm_inp_dim  #39
        self.lstm_hidden_dim = 512
        self.lstm_layers = 4
        self.batch_norm = True
                
        self.bilstm = nn.LSTM(input_size=self.lstm_inp_dim, hidden_size=self.lstm_hidden_dim, num_layers=self.lstm_layers,
                              bidirectional=True,batch_first=True)
        
    def forward(self, input):
        ys, _ = self.bilstm(input)  #1024
        return ys


# # Matching Network
class MatchingNetwork(nn.Module):
    def __init__(self, n, k, q, num_input_channels, lstm_input_size,use_cuda=False):
 
        super(MatchingNetwork, self).__init__()
        self.n = n
        self.k = k
        self.q = q
        self.num_input_channels = num_input_channels
        self.lstm_input_size = lstm_input_size
        self.encoder_g = few_shot_lstm_encoder(self.lstm_input_size)     # LSTM encoder for SS , emb_size = 1024
        self.encoder_f = few_shot_lstm_encoder(self.lstm_input_size)     # LSTM encoder for batch , emb_size = 1024
        
        isCudaAvailable = torch.cuda.is_available()
        self.use_cuda = use_cuda
        
        if isCudaAvailable & self.use_cuda:
            self.device=torch.device('cuda:1')
        else:
            self.device=torch.device('cpu')

    def forward(self, inputs):
        pass


# # Cosine Similarity computation
class DistanceNetwork(nn.Module):
    def __init__(self,n,k,q,bshot):
        super(DistanceNetwork, self).__init__()
        self.n=n
        self.k=k
        self.q=q
        self.bshot=bshot

    def forward(self, support_set, query_set):
        eps = 1e-10
        similarities = []
        
        f = self.bshot  # total no:of blanks in support set
        sum_support = torch.sum(torch.pow(support_set, 2), 2)
        support_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
        dot_product = query_set.matmul(support_set.permute(0,2,1))
             
        cosine_similarity = dot_product * (support_magnitude.unsqueeze(1))  # for ctc
        cosine_similarity  = torch.cat( [cosine_similarity[:,:, :f].topk(self.n, dim=2).values, cosine_similarity [:,:, f:]], dim=2) # take top n blanks    
        return cosine_similarity

class episodeBuilder:
    def __init__(self, model, n_shot, k_way, b_shot, q_queries,    # changes for handling more blank samples 
                 batch_size,
                 data_train=None,data_val=None,data_test=None,batch_test=None):
        
        self.optimiser = Adam(model.parameters(), lr=1e-3)
        
        with open("index_to_label.pkl",'rb') as f:
            self.label_dict=pickle.load(f)
        with open("label_to_index.pkl",'rb') as f:
            index_dict=pickle.load(f)
        self.ctc_labels = sorted(index_dict)
        
        self.decoder = CTCBeamDecoder(labels=self.ctc_labels,model_path=None,alpha=0,beta=0,cutoff_top_n=40,cutoff_prob=1.0,
                                      beam_width=100,num_processes=4,blank_id=0,log_probs_input=True)         
        self.model=model
        self.n_shot=n_shot
        self.b_shot=b_shot  # changes for handling more blank samples
        self.k_way=k_way
        self.q_queries=q_queries
        self.batch_size=batch_size
        self.batch_test = batch_test
        
        self.dn = DistanceNetwork(n_shot,k_way,q_queries,b_shot)

        self.data_test=data_test
        
        self.device=self.model.device
           
    def each_episode_ctc(self, X,y,batch_query, trainflag=False):
        """Performs a single training episode for a Matching Network.
        # Returns
            loss: Loss of the Matching Network on this task
            y_pred: Predicted class probabilities for the query set on this task
        """
        self.train = trainflag
        
        if self.train:
            # Zero gradients
            self.model.train()
            self.optimiser.zero_grad()
        else:
            self.model.eval()

        self.batch_size=X.size(0)
        
        inputs, input_sizes, targets, target_sizes, utt = batch_query
        inputs = inputs.to(self.device)
        
        X = torch.squeeze(X,2)
        X = X.reshape(X.shape[0],X.shape[1], -1)  #(batch_size,frames,inp_dim)
        
        inputs = torch.squeeze(inputs,2)
        inputs = inputs.reshape(inputs.shape[0],inputs.shape[1], -1)  #(batch_size,frames,inp_dim)
        
        # Embed all samples (f, g - not shared parameters)
        embeddings_X = self.model.encoder_g(X)      # SS
        embeddings_X_ctc = self.model.encoder_f(inputs)  # batch
        
        support = embeddings_X
        queries = embeddings_X_ctc
    
        similarities = self.dn(support_set=support, query_set=queries)
        softmax = nn.LogSoftmax(dim=2)
        attention = self.matching_net_predictions(similarities)
        ypreds = softmax(attention)   # [batch_size , len, num_classes]
        
        ypreds = ypreds.to(self.device, dtype=torch.float32)  
        
        beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(ypreds)    #BATCHSIZE x N_TIMESTEPS x N_LABELS
        pred_index = beam_results[0][0][:out_lens[0][0]].tolist()
        pred_label = [self.label_dict[ele] for ele in pred_index]

        target_label_indices = torch.squeeze(targets,0).tolist()
        target_label = [self.label_dict[ele] for ele in target_label_indices]
  
        per = wer(target_label,pred_label)
        self.test_file.write("\n".join(utt))  
        self.test_file.write('ground_truth : %s\n'%(str(target_label)))
        self.test_file.write('decoded_label : %s\n'%(str(pred_label)))
        self.test_file.write('\nper = %f\n'%(per)) 
        
        return per

    def matching_net_predictions(self, attention):
        """Calculates Matching Network predictions based on equation (1) of the paper.
        """
        q=self.q_queries
        k=self.k_way
        n=self.n_shot
        
        y_preds=[]
        for eachbatch in range(attention.size(0)):
            # Create one hot label vector for the support set
            y_onehot = torch.zeros(k * n, k)
            
            ys = self.create_nshot_task_label(k, n).unsqueeze(-1)       
            y_onehot = y_onehot.scatter(1, ys, 1)
        
            y_pred = torch.mm(attention[eachbatch], y_onehot.to(self.device, dtype=torch.float32))
            y_preds.append(y_pred)
            
        y_preds=torch.stack(y_preds)

        return y_preds
    
    def create_nshot_task_label(self, k, n):
        return torch.arange(0, k, 1 / n).long()    
    
    def run_test_epoch(self, total_test_batches):
        total_per = 0.0
        with tqdm(total=total_test_batches, desc='test batches:', leave=False) as pbar:
            pred_list=[]
            for i,  batch_query in enumerate(self.batch_test):   # to iterate through all utterances in an epoch
                support_data = next(iter(self.data_test))
                X,y,ylabels=support_data
                X=X.to(self.device, dtype=torch.float32)
                y=y.to(self.device, dtype=torch.long)
                per = self.each_episode_ctc(X, y, batch_query, trainflag=False)
                total_per += per
                logging.info("test utterance {} per => {} and and avg per {}".format(i, per,total_per/(i+1)))
                pbar.update(1)
                
        total_per = total_per / total_test_batches
        return total_per

def build_model(n_train, k_train, q_train, fce=False, use_cuda=False):
    num_input_channels=1 
    lstm_input_size = 39 # encoder output   ## check

    model = MatchingNetwork(n_train, k_train, q_train, num_input_channels, lstm_input_size,      # include lstm size
                            use_cuda=use_cuda)

    model=model.to(model.device, dtype=torch.float32)
    return model

def load_model(fpath, n_train, k_train, q_train, use_cuda=False, eval_flag=True):
    model = build_model(n_train, k_train, q_train, use_cuda=use_cuda)
    model.load_state_dict(torch.load(fpath,map_location=model.device))
    model = model.to(model.device, dtype=torch.float32)
    if eval_flag:
        model.eval()
    return model


#now we will Create and configure logger 
logging.basicConfig(filename=config.log_file_path, format='%(asctime)s %(message)s', filemode='w') 

#Let us Create an object 
logger=logging.getLogger() 

#Now we are going to Set the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 
logging.getLogger('matplotlib').setLevel(logging.ERROR)

logging.info('MN-CTC Inference')

N=config.N
q=1;#q=1 is supported as of now
batch_size=1 # batch_size=1 is recommended as of now
logging.info(config.inferenceSupportSet)
logging.info(config.inferenceQuerySet)

test_dataset=SupportDataSet(supportdatafile=config.inferenceSupportSet, nshot=N, nqueries=q, transform=transforms.Compose([stackup()]))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_wrapper)

with open("label_to_index.pkl",'rb') as f:
    index_dict=pickle.load(f)
    K = len(index_dict.keys())
bshots=(K-1)*int(N/2)
logging.info('bshots {}'.format(bshots))

batch_dataset=QueryDataSet(querysetfile=config.inferenceQuerySet, index_dict=index_dict, transform=transforms.Compose([batch_stackup(index_dict)]))                              
batch_loader = DataLoader(batch_dataset, batch_size=batch_size, collate_fn=pad_batch)
total_test_batches = batch_loader.__len__()

logging.info('Loading model={}'.format(config.inference_model))
inference_model=load_model(config.inference_model, N, K, q, use_cuda=config.use_cuda, eval_flag=True)

episode1 = episodeBuilder(inference_model, N, K, bshots, q, batch_size, data_test=test_dataloader, batch_test=batch_loader)
episode1.test_file = open(config.inference_results_file,"w")
        
per = episode1.run_test_epoch(total_test_batches)
logging.info("Per of network {}".format(per*100))
episode1.test_file.write('\nper of network = %f\n'%(per))
episode1.test_file.close()


