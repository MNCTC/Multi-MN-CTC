import pickle
import numpy as np
import logging
import sys

datapath = sys.argv[1]
    
#now we will Create and configure logger 
logging.basicConfig(filename=datapath+"/log", format='%(asctime)s %(message)s', filemode='w') 

#Let us Create an object 
logger=logging.getLogger() 

#Now we are going to Set the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 
logging.getLogger('matplotlib').setLevel(logging.ERROR)

## Support-set preparation
logging.info('SS and Batch preparation for Multilngual MN-CTC framework')

SOURCE_LANGUAGES=["Hindi", "Gujarati", "Marathi"]
SOURCE_FILES={"bilstm_trainSS_dim39.pkl", 
              "test_xz.pkl", "dev_xz.pkl", "train_xz.pkl"}

for source_file in SOURCE_FILES:    
    combined_pickle = {}
    combined_lengths = {}
    filename=""
    for source in SOURCE_LANGUAGES:    
        pickle_file_name = datapath + "/" + source +"/"+ source_file
        logging.info('source={}'.format(pickle_file_name))
        with open(pickle_file_name,'rb') as f1:
            data_dict=pickle.load(f1)
        length_dict = {key: len(value) for key, value in data_dict.items()}
#         print(source, "language data", length_dict)
        for key, value in data_dict.items():
            if key in combined_pickle:
                old_value = combined_pickle[key]
                old_value.extend(value)
                combined_pickle[key] = old_value
                combined_lengths[key] = len(old_value)
                logging.info('repeated key= {},lenght= {}'.format(key,combined_lengths[key]))
            else:
                combined_pickle[key] = value
                combined_lengths[key] = len(value)

        filename+=source[0:3]  
    filename+="_"+source_file
    logging.info('multilingual filename={}'.format(filename))
    
    file = open(datapath + "/" + filename, 'wb')
    pickle.dump(combined_pickle,file)
    file.close()
    
# datapath+="/"    
# # generating MN-CTC related labels
# for source in SOURCE_LANGUAGES:
#     datapath+=source[0:3]

# trainSupportSet = datapath+"_bilstm_trainSS_dim39.pkl"
# logging.info('Labels for {}'.format(trainSupportSet))

# with open(trainSupportSet,'rb') as f1:
#     supportData=pickle.load(f1)
# ALLPHONEMES = sorted(supportData.keys())        
# logging.info('ALLPHONEMES={}'.format(ALLPHONEMES))
    
# index_dict = {"blank": 0, "sil": 1}
# label_dict = {0: "blank", 1: "sil"}
# ctc_labels = ['_','sil']

# i=2
# for letter in ALLPHONEMES:
#     if letter not in ['blank', 'sil']:
#         index_dict[letter] = i
#         label_dict[i] = letter
#         ctc_labels.append(letter)
#         i+=1

# logging.info('mn_index_dict={}'.format(index_dict))
# logging.info('mn_label_dict={}'.format(label_dict))
# logging.info('Available classes={}, Count={}'.format(ctc_labels, len(ctc_labels)))