# -*- coding: utf-8 -*-

datapath = sys.argv[1] + "/"
INFERENCE_LANGUAGE = sys.argv[2] 

# Parameters
seed=25
use_cuda= True
balanced=True

# if balanced:
#     datapath = "./MN_balancedData/"
# else:
#     datapath = "./MN_unbalancedData/"
    
datapath+=INFERENCE_LANGUAGE
inferenceSupportSet = datapath+"/bilstm_trainSS_dim39.pkl"
inferenceQuerySet = datapath+'/test_xz.pkl'

#K-way N-shot
N=20 #matched train and test conditions

model_store_path="./MN_model/Multi_MNCTC_"+str(N)+"shots_episodic_batch10"
log_file_path=model_store_path+"/inferencelog"
inference_model=model_store_path+"/model_53_20_1.pth"
inference_results_file=model_store_path+"/inference_"+INFERENCE_LANGUAGE+".txt"