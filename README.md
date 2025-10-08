# HAR-FM: Towards Customizable Foundation Models for HAR with Wearable Device
This repository contains PyTorch implementations for training and fine-tuning the HARFM proposed in [1].
Below is part of the abstract and overview of the training strategy for reference.

> Foundation models have achieved remarkable success across various domains by learning general representations from raw data, offering a promising paradigm for diverse applications. This concept holds great potential for advancing human activity recognition (HAR), particularly in overcoming challenges associated with collecting large-scale labeled datasets. However, the dynamic nature of HAR tasks, characterized by diverse sensing devices and activity types, results in fragmented datasets that question the feasibility of applying foundation model to this domain. In this work, we propose a novel foundation model training framework that effectively leverages heterogeneous datasets through a two-stage training strategy: (1) self-supervised learning to extract cross-domain sensor patterns, followed by (2) multi-task learning to align representations with semantic contexts.



## Setup Environment

```bash
conda create -n HARFM python=3.10 pytorch=1.13.1 pytorch-cuda=11.7 pip -c pytorch -c nvidia -y
conda activate HARFM
pip install -r requirements.txt
```

## Modules Overview

You may find instructions in each module's `Readme.md` for using the codes.

- `FINETUNE_EXAMPLE`:   Contains the trained model parameters and codes related to fine-tuning the HARFM.
- `TRAIN_EXAMPLE`:      Contains codes related to the proposed two-stage training of an HARFM.
- `dataCenter`:         Contains the preprocessed dataset for demonstration purposes.
    -   Before running the code, you should `unzip dataCenter/WRIST.zip` and place it in the `dataCenter` accordingly.

## Acquiring Datasets and Model Parameters

You may download the required dataset and model parameters from [this link](https://drive.google.com/drive/folders/1j1-3OJHWqi7TkrKC43BByXaG0MR6QXCB?usp=drive_link). 

- `FINETUNE_EXAMPLE/EP1_RESULTS.zip`: Contains the training logs and foundation model parameters from the method proposed and evaluated in the paper.
- `FINETUNE_EXAMPLE/FINETUNE_RESULTS.zip`: Contains corresponding results of the finetuning experiment in the paper.
- `dataCenter/WRIST.zip`: Contains the preprocessed datasets used in this demonstration.


## Contact
If you have any questions or suggestions, feel free to contact:

Minghui Qiu [mqiu585@connect.hkust-gz.edu.cn](mqiu585@connect.hkust-gz.edu.cn)


## Acknowledgements
We sincerely express our gratitude to the excellent open-source repositories:

[LIMU-BERT](https://github.com/dapowan/LIMU-BERT-Public.git)

[wav2vec2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/wav2vec2)

## Citations
If you find it useful, please cite our paper.

```bibtex
@article{qiu2025towards,
  title={Towards Customizable Foundation Models for Human Activity Recognition with Wearable Devices},
  author={Qiu, Minghui and Weng, Cekai and Fan, Mingming and Wu, Kaishun},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={9},
  number={3},
  pages={1--29},
  year={2025},
  publisher={ACM New York, NY, USA}
}
```