#!/bin/bash

# Training multi-lingual MN-CTC  acoustic model and decoding 

stage=0

if [ ! -z $1 ]; then
    stage=$1
fi

if [ ${stage} -le 0 ]; then
    echo "Step 0: Multi-lingual support set and batch preparation..."  
    python MN_dataPrep/CombinePickles.py MN_balancedData || exit 1;
fi

if [ ${stage} -le 1 ]; then
    echo "Step 1: Multilingual MN-CTC acoustic model training..."
    python MN_model/multi_episodic_train.py MN_balancedData || exit 1;
fi

if [ ${stage} -le 2 ]; then
    echo "Step 2: MN-CTC inference on Target language..."
    python MN_model/multi_episodic_inference.py MN_balancedData Marathi  || exit 1;
fi