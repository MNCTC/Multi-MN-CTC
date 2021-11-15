import torch
from torch.utils.data import Dataset
import numpy as np
import random    
import pickle

class batch_stackup(object):
    def __init__(self, index_dict):
        self.index_dict = index_dict
    def __call__(self, sample):
        utt_id, utt_set= sample['utt_id'], sample['utt_data']
       
        X_ctc = []
        
        X_ctc_feat = utt_set['feats']
        zlabels = utt_set['labels']
        zindex = [self.index_dict[ele] for ele in zlabels]

        for i, details in enumerate(X_ctc_feat):
            details = details.reshape(1,details.shape[0])
            X_ctc.append(np.array(details))              #inp_dimension = 39
        
        X_ctc=np.asarray(X_ctc, dtype=np.float32)
        
        return torch.from_numpy(X_ctc), torch.LongTensor(zindex), utt_id
    
class QueryDataSet(Dataset):
    def __init__(self, querysetfile, index_dict, transform=None):
       
        with open(querysetfile,'rb') as f1:         # (x,z) pair
            self.uttdict=pickle.load(f1)

        self.utts=list(self.uttdict.keys()) 
        
        targetset = set(index_dict.keys())
        to_del=[]
        for i in self.utts:
            diff = set(self.uttdict[i]['labels']).difference(targetset) 
            if len(diff) > 0:
                print("ignoring ", i)
                to_del.append(i)
        for i in to_del:
            self.utts.remove(i)
            
        self.transform = transform
        
    def __len__(self):
        length=0
        length = len(self.utts)
        return length

    def __getitem__(self, ix):

        sample = {'utt_id' : self.utts[ix], 'utt_data':self.uttdict[self.utts[ix]]} 
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
def sort_batch(batch_data, input_sizes, zlabel, target_sizes, utt_list):
    """
    Sort a minibatch by the length of the sequences with the longest sequences first
    return the sorted batch targes and sequence lengths.
    This way the output can be used by pack_padded_sequences(...)
    """
    sort_input_sizes, perm_idx = input_sizes.sort(0, descending=True)
    sort_batch_data = batch_data[perm_idx]
    sort_zlabel = zlabel[perm_idx]
    sort_target_sizes = target_sizes[perm_idx]
    sort_utt_list = [utt_list[i] for i in perm_idx]
    return sort_batch_data, sort_input_sizes, sort_zlabel, sort_target_sizes, sort_utt_list

def pad_batch(DataLoaderBatch):
    """
    DataLoaderBatch should be a list of (sequence, target, utterance_id) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest, 
    """
    inputs_max_length = max(x[0].size(0) for x in DataLoaderBatch)
    feat_size1 = DataLoaderBatch[0][0].size(1)
    feat_size2 = DataLoaderBatch[0][0].size(2)
    targets_max_length = max(x[1].size(0) for x in DataLoaderBatch)
    batch_size = len(DataLoaderBatch)
    batch_data = torch.zeros(batch_size, inputs_max_length, feat_size1,feat_size2) 
    zlabel = torch.zeros(batch_size, targets_max_length)
    input_sizes = torch.zeros(batch_size)
    target_sizes = torch.zeros(batch_size)
    utt_list = []

    for x in range(batch_size):
        feature, ctc_label, utt = DataLoaderBatch[x]
        feature_length = feature.size(0) 
        label_length = ctc_label.size(0)
        
        batch_data[x].narrow(0, 0, feature_length).copy_(feature)
        zlabel[x].narrow(0, 0, label_length).copy_(ctc_label)
        input_sizes[x] = feature_length
        target_sizes[x] = label_length
        utt_list.append(utt)
    
    #batch_data=torch.unsqueeze(batch_data,2)  
    #return batch_data.float(), input_sizes.float(), zlabel.long(), target_sizes.long(), utt_list 
    #return sort_batch(batch_data.float(), input_sizes.long(), zlabel.long(), target_sizes.long())
    return sort_batch(batch_data.float(), input_sizes.long(), zlabel.long(), target_sizes.long(), utt_list) 