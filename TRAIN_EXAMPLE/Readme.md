# This is an example of training HARFM.

Note: This example reproduces the training process with the released dataset (../dataCenter/WRIST/2019-WISDM) and only serves as a demonstration. 


# How to run?
1. Self-supervised Learning to model the pattern with unlabelled data.

```bash
cd TRAIN_1_SSL
python train_ssl.py --TRAIN_SOURCE --PFM_VERSION pfm34 --LEARNING_RATE_DEFACTOR 2 --AMSGRAD --CODEBOOK_NUM_GROUPS 6 --CODEBOOK_NUM_VARS 4 --LOSS_TYPE 0
```
2. Multi-task Learning to align the representations with the semantics.

```bash
cd TRAIN_2_FM
python train_fm.py --TRAIN_SOURCE --loss_style 0
```