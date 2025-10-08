

## Description
This repository serves as a demo to run the first stage of self-supervised learning with the released dataset.

## How to run?
Activate the environment and run the following command under this folder,
'''
python train_ssl.py --TRAIN_SOURCE --PFM_VERSION pfm34 --LEARNING_RATE_DEFACTOR 2 --AMSGRAD --CODEBOOK_NUM_GROUPS 6 --CODEBOOK_NUM_VARS 4 --LOSS_TYPE 0
'''