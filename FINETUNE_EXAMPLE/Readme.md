# Customizable Fine-tuning

In this example, we demonstrate how to fine-tune the HARFM model for evaluation purposes.

## File Overview
- `EP1_RESULTS.zip`: Contains the foundation model parameters proposed and evaluated in the paper
- `results*/` folders: Contain fine-tuned model parameters and documentation for:
  - Specific folds (from 5-fold experiments)
  - Specific input window lengths (e.g., `_w1000` for 1000ms windows)

## How to Run
First, unzip the foundation model:
```bash
unzip EP1_RESULTS.zip
```

### A. Fine-tuning Scripts
#### Case I: Linear Adapter
```bash
python s1_finetune.py --encoder_layer_num 0 --TAR_FOLD 0 --EXP_NAME test --TEST_TIME_WINDOW 800
```
Output Directory
```text
results_w800/test_linear_finetuneAll/*/0/
```

#### Case II: 3-layer CNN Adapter
```bash
python s1_finetune.py --encoder_layer_num 3 --TAR_FOLD 0 --EXP_NAME test --TEST_TIME_WINDOW 800
```
Output Directory
```text
results_w800/test_finetuneAll/*/0/
```

### B. Testing Pre-trained Models
To test the fine-tuned model from the paper (1000ms window, fold 0):

1. Unzip results
```bash
unzip FINETUNE_RESULTS.zip
```

2. Run evaluation
```bash
python s2_test.py --EXP_NAME postTrain_finetuneAll --TEST_TIME_WINDOW 1000 --TAR_FOLD 0
```
Output file:
```text
results_w1000/postTrain_finetuneAll/*/0/test-logs/test_log_YYYYMMDD_HHmmss.csv
```

### C. View Results
To display fold-averaged results:
```bash
python s3_dsp_results.py
```
