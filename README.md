# Multi-lingual MN-CTC

#### Environments:
*   Kaldi toolkit
*   pytorch 1.4.0

### Input Data
Training and Inference support-set and batch utterance files of Hindi, Gujarati and Marathi can be downloaded from here :
https://drive.google.com/drive/folders/11_ruvvWcBgRgscSLyRPwxqPwVT4RvEzV?usp=sharing

### 1. Feature Extraction
39 dimension MFCC (25 ms frames shifted by 10ms each time) using Kaldi toolkit 

### 2. Support set preparation

Train Bi-LSTM-CTC using train utterances of source corpus. Trained model is used to generate the support set files.
In multi-MN-CTC, support set files of individual languages are combined using a common phone-lable set.

### 3. MN-CTC 

#### 1. Input files  : 

1. Train support set (S) generated from Train utterances of multiple source language.
2. Batch utterances : (x,z) pair - Train utterance and ground truth transcript of different source language.
3. Inference support set (S') generated from validation utterances of target langauge
4. Query utterance : Test utterance of target language

#### 2. MN Training

End-to-end MN training using CTC loss function. The network consists of 2 encoders 'g' and 'f' to embed the support-set samples and batch utterance.

MN consists of two encoders g and f to embed the support samples and batch utterance.
1. encoder 'g' - to embed support set frames - fed as an spectrographic patch (39x11) to a 3-layer CNN
2. encoder 'f' - uses bi-LSTM to map utterances to 256 dimension space.

#### 2. MN Inference
Decode the test utterances of target language using the multi-lingual MN-CTC model.
