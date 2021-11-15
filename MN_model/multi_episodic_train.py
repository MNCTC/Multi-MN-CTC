#!/usr/bin/env python
# coding: utf-8

# Uncoupled MN_CTC (LSTM - 4 layer, inp_dim = 39 and embedding dim = 1024)

import train_Configuration as config
from episodic_dataloader.ctc_batch_loader import QueryDataSet, batch_stackup, pad_batch

import sys
import os
from datetime import datetime
import random
import pickle
from torch.optim import Adam
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torch
from torchvision import transforms
import torchvision
from ctcdecode import CTCBeamDecoder
from jiwer import wer
from tqdm.notebook import tqdm
import editdistance as ed
import matplotlib.pyplot as plt
import logging


EPSILON = 1e-8
seed=25
torch.manual_seed(seed)
np.random.seed(seed)

# embedding network
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

# Matching Network
class MatchingNetwork(nn.Module):
    def __init__(self, n, k, q, lstm_input_size=429,use_cuda=False):
        """Creates a Matching Network as described in Vinyals et al.
        """
        super(MatchingNetwork, self).__init__()
        self.n = n
        self.k = k
        self.q = q
        self.lstm_input_size = lstm_input_size
        self.encoder_f = few_shot_lstm_encoder(self.lstm_input_size)  # include lstm size
        self.encoder_g = few_shot_lstm_encoder(self.lstm_input_size)
        
        isCudaAvailable = torch.cuda.is_available()
        self.use_cuda = use_cuda
        
        if isCudaAvailable & self.use_cuda:
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')

    def forward(self, inputs):
        pass

# Distance Network
class DistanceNetwork(nn.Module):
    def __init__(self,n,k):
        super(DistanceNetwork, self).__init__()
        self.n=n
        self.k=k

    def forward(self, support_set, query_set, bshot):
        eps = 1e-10
        similarities = []
        
        f = bshot  # total no:of blanks in support set
        sum_support = torch.sum(torch.pow(support_set, 2), 2)
        support_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
        dot_product = query_set.matmul(support_set.permute(0,2,1))
             
        cosine_similarity = dot_product * (support_magnitude.unsqueeze(1))  # for ctc
        cosine_similarity  = torch.cat( [cosine_similarity[:,:, :f].topk(self.n, dim=2).values, cosine_similarity [:,:, f:]], dim=2) # take top n blanks    
        return cosine_similarity

# Support-set preparation
class SupportDataSet(nn.Module):
    def __init__(self, supportdata, nshot, index_dict, label_dict): 
        super(SupportDataSet, self).__init__()
        
        self.data=supportdata
        self.nshot=nshot
        self.index_dict = index_dict
        self.label_dict = label_dict
        
    def forward(self, zlabels, bshot):
        S_set = []
        for classno in zlabels:
            new_details=[]
            eachphone = self.label_dict[classno]
            if eachphone == 'blank':  
                blank_details=self.data[eachphone]
                if len(blank_details) < bshot:
                    raise Exception("Data Error in T set:no of blank samples less than",bshot) #bshot : no:of blank samples
                tmp=random.sample(blank_details, bshot)
            else:
                details=self.data[eachphone]
                tmp = []
                if(len(details)>=self.nshot):
                    tmp=random.sample(details,self.nshot)
                else:
                    while((len(tmp)+len(details))<self.nshot):
                        tmp.extend(details)

                    tmp.extend(random.sample(details,self.nshot-len(tmp)))

            for item in tmp:
                new_details.append([self.index_dict[eachphone], item])
            S_set.append(new_details)
        
        XS,yS = [],[]
        for i, details in enumerate(S_set):
            for it in details:
                XS.append(np.array(it[1]))  # inp dim 39
                yS.append(it[0])
       
        XX=np.asarray(XS, dtype=np.float32)
        X=torch.from_numpy(XX)
        X=torch.unsqueeze(X,0)
        
        y=torch.from_numpy(np.asarray(yS, dtype=np.long))
        
        return X,y

# Epoch Builder
class episodeBuilder:
    def __init__(self, model, n_shot, k_way, supportdata, index_dict, label_dict, batch_size=1,
                 data_train=None,batch_train=None,batch_val=None,batch_test=None):
        
        self.ctc_loss = nn.CTCLoss(reduction='sum')
        self.optimiser = Adam(model.parameters(), lr=1e-3)
        
        self.index_dict = index_dict
        self.label_dict = label_dict
        self.ctc_labels = sorted(self.index_dict)
        
        self.decoder = CTCBeamDecoder(labels=self.ctc_labels,model_path=None,alpha=0,beta=0,cutoff_top_n=40,cutoff_prob=1.0,
                                      beam_width=100,num_processes=4,blank_id=0,log_probs_input=True)  

        self.model=model
    
        self.n_shot=n_shot
        self.k_way=k_way
        self.batch_size=batch_size
        
        self.dn = DistanceNetwork(n_shot,k_way)
        self.SSdata = SupportDataSet(supportdata, n_shot, self.index_dict, self.label_dict )

        self.batch_train = batch_train
        self.batch_val = batch_val
        self.batch_test = batch_test
        
        self.device=self.model.device
            
    def compute_wer(self, index, input_sizes, targets, target_sizes):
        batch_errs = 0
        batch_tokens = 0
        for i in range(len(index)):
            label = targets[i][:target_sizes[i]]
            pred = []
            for j in range(len(index[i][:input_sizes[i]])):
                if index[i][j] == 0:
                    continue
                if j == 0:
                    pred.append(index[i][j])
                if j > 0 and index[i][j] != index[i][j-1]:
                    pred.append(index[i][j])
            batch_errs += ed.eval(label, pred)
            batch_tokens += len(label)
        return batch_errs, batch_tokens

    def each_episode_ctc(self, batch_query, trainflag=True):
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
        
        inputs, input_sizes, targets, target_sizes, utt = batch_query
        inputs = inputs.to(self.device)
        input_sizes = input_sizes.to(self.device)
        target_sizes = target_sizes.to(self.device)
        ys = [y[y != 0] for y in targets]  # parse padded ys
        ys_true = torch.cat(ys).cpu().long()  # batch x olen
        
        zlabels = torch.unique(ys_true)      # get unique batch labels
        bshot = len(zlabels) * int(self.n_shot/2)  #get bshot value in accordance to phoneme classes in batch
        blank_labels = torch.zeros(1).long() 
        zlabels =  torch.cat((blank_labels,zlabels),dim=0)  #Pprime labels
        
        SS_all_labels = torch.arange(0, self.k_way).view(-1, 1).repeat(1, self.n_shot).view(1, -1).long()   # all labels in SS
        SS_batch_labels = SS_all_labels.view(1, -1).eq(zlabels.view(-1, 1)).sum(0) # set SS labels in batch/query utt as 1 and other labels as 0
        
        # get support set samples
        X,y = self.SSdata(zlabels.numpy(),bshot)
        X=X.to(self.device, dtype=torch.float32)  #(batch size, frames, dimension)
        y=y.to(self.device, dtype=torch.long)  
        
        inputs = inputs.reshape(inputs.shape[0],inputs.shape[1], -1)  #(batch_size,frames,inp_dim)
        
        # Embed all samples (f, g - not shared parameters)
        embeddings_X = self.model.encoder_g(X)      # SS
        embeddings_X_ctc = self.model.encoder_f(inputs)  # batch
        
        support = embeddings_X
        queries = embeddings_X_ctc
        similarities = self.dn(support_set=support, query_set=queries, bshot=bshot) # tensor B, subset of A (batchsize, batchframes, Pprime*Q)
        similarities_appended = torch.zeros(similarities.shape[0], similarities.shape[1], SS_batch_labels.shape[0]) # initialize a 0 tensor A-(batchsize,batchframes,P*Q)
        similarities_appended = similarities_appended.to(self.device, dtype=torch.float32)  # move to cuda
        pos = (SS_batch_labels != 0).nonzero().view(-1) # insert positions of B into A 
        similarities_appended[:,:,pos]=similarities #insert B into relevant location given by pos in A
        
        softmax = nn.LogSoftmax(dim=2)
        attention = self.matching_net_predictions(similarities_appended, SS_batch_labels)
        ypreds = softmax(attention)
        ypreds = ypreds.transpose(0,1)  # [len, batch_size , num_classes]
        ypreds = ypreds.to(self.device, dtype=torch.float32)   
        
        out_len, b_size, _ = ypreds.size()
        ys_true = ys_true.long().to(self.device)
                          
        loss = self.ctc_loss(ypreds, ys_true, input_sizes, target_sizes)
        loss /= b_size
         
        values, indices = torch.max(ypreds, dim=-1)
            
        batch_errs, batch_tokens = self.compute_wer(indices.transpose(0,1).cpu().numpy(), input_sizes.cpu().numpy(), 
                                                    targets.cpu().numpy(), target_sizes.cpu().numpy())
        
        error = batch_errs/batch_tokens
        return loss, error

    def each_episode_ctc_test(self, batch_query, trainflag=False):
        """Performs a testing for a Matching Network.
        # Returns
            PER: PER of the Matching Network on this task
        """
        self.train = trainflag
        
        if self.train:
            # Zero gradients
            self.model.train()
            self.optimiser.zero_grad()
        else:
            self.model.eval()
        
        inputs, input_sizes, targets, target_sizes, utt = batch_query
        inputs = inputs.to(self.device)
        input_sizes = input_sizes.to(self.device)
        target_sizes = target_sizes.to(self.device)
        ys = [y[y != 0] for y in targets]  # parse padded ys
        ys_true = torch.cat(ys).cpu().long()  # batch x olen
        
        zlabels = torch.unique(ys_true)      # get unique batch labels
        bshot = len(zlabels) * int(self.n_shot/2)  #get bshot value in accordance to phoneme classes in batch
        blank_labels = torch.zeros(1).long() 
        zlabels =  torch.cat((blank_labels,zlabels),dim=0)  #Pprime labels
        
        SS_all_labels = torch.arange(0, self.k_way).view(-1, 1).repeat(1, self.n_shot).view(1, -1).long()   # all labels in SS
        SS_batch_labels = SS_all_labels.view(1, -1).eq(zlabels.view(-1, 1)).sum(0) # set SS labels in batch/query utt as 1 and other labels as 0
        
        # get support set samples
        X,y = self.SSdata(zlabels.numpy(),bshot)
        X=X.to(self.device, dtype=torch.float32)  #(batch size, frames, dimension)
        y=y.to(self.device, dtype=torch.long)  
        
        inputs = inputs.reshape(inputs.shape[0],inputs.shape[1], -1)  #(batch_size,frames,inp_dim)

        # Embed all samples (f, g - not shared parameters)
        embeddings_X = self.model.encoder_g(X)      # SS
        embeddings_X_ctc = self.model.encoder_f(inputs)  # batch
        
        support = embeddings_X
        queries = embeddings_X_ctc
        similarities = self.dn(support_set=support, query_set=queries, bshot=bshot) # tensor B, subset of A (batchsize, batchframes, Pprime*Q)
        similarities_appended = torch.zeros(similarities.shape[0], similarities.shape[1], SS_batch_labels.shape[0]) # initialize a 0 tensor A-(batchsize,batchframes,P*Q)
        similarities_appended = similarities_appended.to(self.device, dtype=torch.float32)  # move to cuda
        pos = (SS_batch_labels != 0).nonzero().view(-1) # insert positions of B into A 
        similarities_appended[:,:,pos]=similarities #insert B into relevant location given by pos in A
        
        softmax = nn.LogSoftmax(dim=2)
        attention = self.matching_net_predictions(similarities_appended, SS_batch_labels)
        ypreds = softmax(attention)
        ypreds = ypreds.transpose(0,1)  # [len, batch_size , num_classes]
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
        self.test_file.write('\nper = %f\n'%(per*100))
        
        return per
    
    def matching_net_predictions(self, attention, not_batch_labels):
        """Calculates Matching Network predictions based on equation (1) of the paper.
        """
        k=self.k_way
        n=self.n_shot
        
        y_preds=[]
        for eachbatch in range(attention.size(0)):
            # Create one hot label vector for the support set
            y_onehot = torch.zeros(k * n, k)
            ys = self.create_nshot_task_label(k, n).unsqueeze(-1)       
            y_onehot = y_onehot.scatter(1, ys, 1)
            y_onehot_batch = (not_batch_labels.view(-1,1)) * y_onehot  # ignore classes not in train batch query
            y_pred = torch.mm(attention[eachbatch], y_onehot_batch.to(self.device, dtype=torch.float32))
            y_preds.append(y_pred)
            
        y_preds=torch.stack(y_preds)
        return y_preds

    def create_nshot_task_label(self, k, n):
        return torch.arange(0, k, 1 / n).long()    
    
    def run_training_epoch(self, total_train_batches):
        total_loss = 0.0
        total_error = 0.0
    
        with tqdm(total=total_train_batches, desc='train', leave=False) as pbar1:    
            for i,  batch_query in enumerate(train_batch_loader): # to iterate through all utterances in an epoch
                               
                loss, err = self.each_episode_ctc(batch_query, trainflag=True)

                total_loss += loss
                total_error += err
                
                loss.backward()
                #clip_grad_norm_(self.model.parameters(), 1)
                self.optimiser.step()
        
                pbar1.update(1)
        
        total_loss = total_loss / total_train_batches
        total_error = total_error / total_train_batches
         
        return total_loss, total_error
        
    def run_val_epoch(self, total_val_batches):
        total_loss = 0.0
        total_error = 0.0
               
        with tqdm(total=total_val_batches, desc='val', leave=False) as pbar1:
            for i,  batch_query in enumerate(self.batch_val):  # to iterate through all utterances in an epoch
          
                loss, err = self.each_episode_ctc(batch_query, trainflag=False)

                total_loss += loss.data
                total_error += err
                
                pbar1.update(1)

        total_loss = total_loss / total_val_batches
        total_error = total_error / total_val_batches

        return total_loss, total_error

    def run_test_epoch(self, total_test_batches):
        total_per = 0.0
        
        with tqdm(total=total_test_batches, desc='test batches:', leave=False) as pbar:
            for i,  batch_query in enumerate(self.batch_test):   # to iterate through all utterances in an epoch
                per = self.each_episode_ctc_test(batch_query, trainflag=False)
                total_per += per
                pbar.update(1)
                
            total_per = total_per / total_test_batches
            return(total_per)
            
    def save_model(self, tepochs, fpath):
        fpath=fpath[:-4]+'_'+str(self.k_way)+'_'+str(self.n_shot)+ '_'+str(tepochs)+fpath[-4:]
        torch.save(self.model.state_dict(), fpath)
        return fpath


def build_model(n_train, k_train, q_train, fce=False, use_cuda=False):
    lstm_input_size = 39 # encoder output   ## check

    model = MatchingNetwork(n_train, k_train, q_train, lstm_input_size,      # include lstm size
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

# get the phone classes from support set file.
supportdatafile = config.trainSupportSet
with open(supportdatafile,'rb') as f1:
    supportdata=pickle.load(f1)
labels = sorted(supportdata.keys())
index_dict = {"blank": 0, "sil": 1}
label_dict = {0: "blank", 1: "sil"}
i=2
for phn in labels:
    if phn not in ['blank', 'sil']:
        index_dict[phn] = i
        label_dict[i] = phn
        i+=1
logging.info("index_dict {}".format(index_dict))
logging.info("label_dict {}".format(label_dict))

#P way Q Shot..
P = len(index_dict.keys())
Q = config.Q
q=1
batch_size=config.batch_size 

# dataloader
train_batch_dataset=QueryDataSet(querysetfile=config.trainQuerySet, index_dict=index_dict, transform=transforms.Compose([ batch_stackup(index_dict)]))
train_batch_loader = DataLoader(train_batch_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch) 

val_batch_dataset=QueryDataSet(querysetfile=config.devQuerySet, index_dict=index_dict, transform=transforms.Compose([ batch_stackup(index_dict)]))     
val_batch_loader = DataLoader(val_batch_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)

test_batch_dataset=QueryDataSet(querysetfile=config.testQuerySet, index_dict=index_dict, transform=transforms.Compose([ batch_stackup(index_dict)]))       
test_batch_loader = DataLoader(test_batch_dataset, batch_size=1, shuffle=True, collate_fn=pad_batch)

# load or build model
old_epochs = config.prev_model_epochs

if old_epochs!=0:
    model = load_model(config.prev_model_path, Q, P, q, use_cuda=True, eval_flag=False)
else:
    model = build_model(Q, P, q, use_cuda=config.use_cuda)

model = model.to(model.device, dtype=torch.float32)
      
epochs = config.epochs

total_train_batches = train_batch_loader.__len__() 
total_val_batches = val_batch_loader.__len__()

episode = episodeBuilder(model, Q, P, 
                        supportdata=supportdata,
                        index_dict = index_dict,
                        label_dict = label_dict,
                        batch_size=batch_size,
                        batch_train=train_batch_loader,
                        batch_val=val_batch_loader,
                        batch_test=test_batch_loader)

logdir=config.model_store_path

# Create the output folder
try:
    os.stat(logdir)
except:
    os.makedirs(logdir)

#now we will Create and configure logger 
logging.basicConfig(filename=config.log_file_path, format='%(asctime)s %(message)s', filemode='w') 

#Let us Create an object 
logger=logging.getLogger() 

#Now we are going to Set the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 
logging.getLogger('matplotlib').setLevel(logging.ERROR)

logging.info('Multi-MN-CTC training')
logging.info('batch_size={}'.format(batch_size))
logging.info('Total epochs={}'.format(config.epochs))
logging.info('model_store_path={}'.format(logdir))
logging.info('model={}'.format(episode.model))

loss, acc, val_loss, val_acc = [], [], [], []

start_time = datetime.now()

with tqdm(total=epochs, desc='epochs', leave=False) as pbar:
    for e in range(epochs):
        logging.info('Start training epochs={}'.format(e))
        epoch_start_time = datetime.now()
        episode.epoch_cnt = e+old_epochs
        total_c_loss, total_error = episode.run_training_epoch(total_train_batches)
        loss.append(total_c_loss); acc.append(total_error)
        total_val_c_loss, total_val_error = episode.run_val_epoch(total_val_batches)
        val_loss.append(total_val_c_loss); val_acc.append(total_val_error)

        logging.info("Epoch {}: train: [loss-{:.6f} error-{:.6f} ], val: [loss-{:.6f} error-{:.6f}]".                 
              format(e+old_epochs, total_c_loss.item(), total_error,                 
                     total_val_c_loss.item(), total_val_error))
        
        episode.train_file = open(logdir+'/train_results.txt',"a")
        episode.train_file.write("Epoch {}: train: [loss-{:.6f} error-{:.6f} ], val: [loss-{:.6f} error-{:.6f}]\n".                 
              format(e+old_epochs, total_c_loss.item(), total_error,                 
                     total_val_c_loss.item(), total_val_error))
        pbar.update(1)
        episode.train_file.close()
        
        end_time = datetime.now()
        logging.info('Epoch Training time {}'.format(end_time-start_time))
        # save model
        modelpath = episode.save_model(e+old_epochs,fpath= logdir + '/model.pth')
        
logging.info('Total Training time {}'.format(end_time-start_time))
logging.info('Epochs: {}, No. of batches in train: {}, No. of batches in val: {}, Each batch size: {}'.format(
                epochs, total_train_batches, total_val_batches, batch_size))
logging.info('Training completed...')

## plots
import matplotlib.pyplot as plt
plt.plot(loss,label='training')
plt.plot(val_loss,label='validation')
plt.ylabel("Loss",fontsize=16)
plt.xlabel("Epoch",fontsize=16)
plt.title("Evolution of the loss function",fontsize=16)
plt.legend(fontsize=16)
plt.savefig(logdir + "/CTC_loss.png")

plt.figure()
plt.plot(acc, label='training')
plt.plot(val_acc,label='validation')
plt.ylabel("Error",fontsize=16)
plt.xlabel("Epoch",fontsize=16)
plt.title("Evolution of the PER function",fontsize=16)
plt.legend(fontsize=16)
plt.savefig( logdir+ "/error.png")


logging.info('Testing started...')
total_test_batches = test_batch_loader.__len__()
episode.test_file = open(logdir + '/test_results.txt',"w")
per = episode.run_test_epoch(total_test_batches)
logging.info("Per of network {} ".format(per*100))
episode.test_file.write('\nper = %f\n'%(per*100))
episode.test_file.close()
logging.info('Testing completed...')