from torch.utils.data import Dataset

import numpy as np
import pickle
import random
import torch
import logging

class stackup(object):
    def __call__(self, support_set):
        X, y, yLabels = [],[],[]
        for i, details in enumerate(support_set):
            for it in details:
                X.append(np.array(it[2]))
                y.append(it[1])
                yLabels.append(it[0])
       
        return X,y,yLabels

class SupportDataSet(Dataset):
    def __init__(self, supportdatafile, nshot, nqueries, transform=None): 
        with open(supportdatafile,'rb') as f1:
            self.data=pickle.load(f1)
        self.labels = sorted(self.data.keys())
        logging.info("support set labels {}".format(self.labels))
        self.form_mappings()    
        self.kway=len(self.labels)
        logging.info("kway {}".format(self.kway))
        self.nshot=nshot
        self.bshot=(self.kway-1)*int(self.nshot/2)
        self.nqueries = nqueries
        
        self.transform = transform

    def form_mappings(self):    
        self.L_label_to_index = {"blank": 0, "sil": 1}
        self.L_index_to_label = {0: "blank", 1: "sil"}
        i=2
        for phn in self.labels:
            if phn not in ['blank', 'sil']:
                self.L_label_to_index[phn] = i
                self.L_index_to_label[i] = phn
                i+=1

        logging.info("label to index dict {}".format(self.L_label_to_index))
        logging.info("index to label dict {}".format(self.L_index_to_label))
        file = open("label_to_index.pkl", 'wb')
        pickle.dump(self.L_label_to_index, file)
        file.close()
        file = open("index_to_label.pkl", 'wb')
        pickle.dump(self.L_index_to_label, file)
        file.close()

    def __len__(self):
        length=0
        for ltr in self.letters:
            if(ltr in self.data):
                length += len(self.data[ltr])
        return length
        
    def __getitem__(self, ix):
        S_set = []
        for classno in range (len(self.letters)):
            new_details=[]
            eachLetter = self.L_index_to_label[classno]  
            if eachLetter == 'blank':  
                blank_details=self.data[eachLetter]
                if len(blank_details) < self.bshot:
                    raise Exception("Data Error in T set:no of blank samples less than",self.bshot) #bshot : no:of blank samples
                tmp=random.sample(blank_details,self.bshot)
            else:
                if(eachLetter in self.data):                    
                    details=self.data[eachLetter]
                    tmp = []
                    if(len(details)>=self.nshot):
                        tmp=random.sample(details,self.nshot)
                    else:
                        while((len(tmp)+len(details))<self.nshot):
                            tmp.extend(details)

                        tmp.extend(random.sample(details,self.nshot-len(tmp)))

            for item in tmp:
                new_details.append([eachLetter, self.L_label_to_index[eachLetter], item])
            S_set.append(new_details)
        
        if self.transform:
            S_set = self.transform(S_set)

        return S_set

def collate_wrapper(batch):
    X_set=[]
    y_set=[]
    ylabels_set=[]
    
    for i, item in enumerate(batch):
        X,y,ylabels=item
        X_set.append(X)
        y_set.append(y)
        ylabels_set.append(ylabels)
    
    XX=np.asarray(X_set, dtype=np.float32)
    X=torch.from_numpy(XX)
    X=torch.unsqueeze(X,2)
        
    y=torch.from_numpy(np.asarray(y_set, dtype=np.long))
    ylabels=ylabels_set
    batch=[X,y,ylabels]
    
    return batch
