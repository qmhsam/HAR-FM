
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, accuracy_score

from models import WearablePretrainedEncoder, grab_avgpool_classifier, grab_avgpool_classifier_linear, FeatureWrapper, SINGLE_CHANNEL_CLASSIFIER
from utils import *
from tqdm import tqdm
from sys import getsizeof as getsize
import csv, json, os
from datetime import datetime

import torch.nn.functional as F

import argparse 
parser = argparse.ArgumentParser(description='tbc')
parser.add_argument('--TAR_FOLD', default=0, type=int, help='k-fold setting', choices=[0, 1, 2, 3, 4])

parser.add_argument('--TEST_TIME_WINDOW', default=1000, type=int, choices=[800, 1000, 2000, 3000, 5000, 6000])
parser.add_argument('--EXP_NAME',  default="finetuning-undefined", type=str)

parser.add_argument('--placement', default="WRIST", type=str, choices=["WRIST", "POCKET", "UpARM"])


args_global = parser.parse_args()

##########################################

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

############################################

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # return torch.device('cpu')

def test_model(model, model_path, test_loader, device, log_dir='logs'):
    # Initialize the model

    # Load the state dictionary
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.to(device)
    model.eval()

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'test_log_{timestamp}.csv')

    # Open the log file and write the header
    with open(log_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user-dev-option', "Accuracy", 'Balanced Accuracy', 'Macro F1-Score', 'Precision'])

    # Containers to store predictions, ground truth, and metadata
    all_predictions = []
    all_targets = []
    all_meta_info = []

    tempRst = {}
    with torch.no_grad():
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target, meta_info) in enumerate(tqdm(test_loader, desc='Testing')):
            data, target = data.to(device), target.to(device)
            target = target - 1
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            # Calculate the number of correct predictions
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)

            # Store predictions, ground truth, and metadata
            metaInfo_ = list(zip(meta_info["_sampleSource"][0], meta_info["_sampleSource"][1]))
            predicted_ = predicted.tolist()
            target_  = target.tolist()
            all_predictions.extend(predicted_)
            all_targets.extend(target_)
            all_meta_info.extend(metaInfo_)

            for i, (fName, ind) in enumerate(metaInfo_):
                name_ = fName.split(os.sep)[-2]
                if name_ not in tempRst:
                    tempRst[name_] = []
                tempRst[name_] += [(predicted_[i], target_[i])]
            
            # Log the test loss, balanced accuracy, macro F1-score, and precision for each batch
            # balanced_acc = balanced_accuracy_score(target.cpu().numpy(), predicted.cpu().numpy())
            # macro_f1 = f1_score(target.cpu().numpy(), predicted.cpu().numpy(), average='macro')
            # precision = precision_score(target.cpu().numpy(), predicted.cpu().numpy(), average='macro')
    for name_ in tempRst:
        predicted_, target_ = zip(*tempRst[name_])
        acc = accuracy_score(target_, predicted_)
        balanced_acc = balanced_accuracy_score(target_, predicted_)
        macro_f1 = f1_score(target_, predicted_, average='macro')
        precision = precision_score(target_, predicted_, average='macro')

        with open(log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name_, acc, balanced_acc, macro_f1, precision])

    average_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples * 100

    overall_acc = accuracy_score(all_targets, all_predictions)
    # Compute and log the overall balanced accuracy, macro F1-score, and precision
    overall_balanced_acc = balanced_accuracy_score(all_targets, all_predictions)
    overall_macro_f1 = f1_score(all_targets, all_predictions, average='macro')
    overall_precision = precision_score(all_targets, all_predictions, average='macro')

    with open(log_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Overall", overall_acc, overall_balanced_acc, overall_macro_f1, overall_precision])
    print(f'Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
    print(f'Balanced Accuracy: {overall_balanced_acc:.4f}, Macro F1-Score: {overall_macro_f1:.4f}, Precision: {overall_precision:.4f}')
    print(f'Test complete. Log saved to {log_file}')

    # Save the predictions, ground truth, and metadata to a file
    output_file = os.path.join(log_dir, f'test_results_{timestamp}.pth')
    torch.save({'predictions': all_predictions, 'targets': all_targets, 'meta_info': all_meta_info}, output_file)
    print(f'Test results saved to {output_file}')

    return average_loss, accuracy, overall_balanced_acc, overall_macro_f1, overall_precision

if __name__ == "__main__":
    
    # DataLoader
    # window settings
    # global windows settings for each datasets
    timeWindowSetting = {
        "20hz":     (4, 200),
        "30hz":     (6, 200),
        "50hz":     (10, 200),
        "100hz":    (20, 200)
    } 

    # In 200 ms on fine-grained task
    # 2012-OPPORTUNITY  6
    # 2012-PAMAP2       20
    # 2012-realdisp     10
    # 2014-mHealth      10
    # 2014-shoaib       10
    # 2015-HHAR         41
    # 2019-WISDM        5
    # 2020-capture24    20
    # 2020-GOTOV        18

    # find the nearest settings
    #   
    dataset_to_window = {
        # "2012-OPPORTUNITY": "30hz",
        # "2012-PAMAP2":      "100hz",
        # "2012-realdisp":    "50hz",
        "2014-mHealth":     "50hz",
        # "2014-shoaib":      "50hz",
        # "2015-HHAR":        "100hz", # downsample from 200hz to 100hz
        # "2019-WISDM":       "30hz",  # "upsample from 25hz to 30hz"
        # "2020-capture24":   "100hz",
        # "2020-GOTOV":       "100hz",  # "upsample from 88hz to 100hz"
    }

    
    # dataset_to_window = {
    #     # "2012-OPPORTUNITY": "30hz",
    #     "2012-PAMAP2":      "30hz",
    #     "2012-realdisp":    "30hz",
    #     "2014-mHealth":     "30hz",
    #     "2014-shoaib":      "30hz",
    #     "2015-HHAR":        "30hz", # downsample from 200hz to 100hz
    #     # "2019-WISDM":       "30hz",  # "upsample from 25hz to 30hz"
    #     "2020-capture24":   "30hz",
    #     "2020-GOTOV":       "30hz",  # "upsample from 88hz to 100hz"
    # }

    # Four customized datasets in future use case.
    #   1. "2019-mid_air gesture"
    #   2. "2020-WHAR"
    #   3. "2020-Taprint"
    #   4. "2023-aMaze"
    
    # static settings
    #   base settings for @DEFAULT_GRAINED_TASK
    #       200 ms time windows [TEST_SAMPLE_TIME_WINDOW]
    
    #   test procedure & results
    #       PLACEMENT   = ["ANKLE", "BACK", "CHEST", "KNEE", "LoARM", "LoKNEE", "POCKET", "UpARM", "UpKNEE", "WAIST", "WRIST"]
    #       DATASET     = os.listdir(PLACEMENT)
    
    # dataset_to_batchsize = { # to 
    #     "2012-OPPORTUNITY": 16,
    #     "2012-PAMAP2":      32,
    #     "2012-realdisp":    16,
    #     "2014-mHealth":     16,
    #     "2014-shoaib":      16,
    #     "2015-HHAR":        16,     # downsample from 200hz to 100hz
    #     "2019-WISDM":       16,     # "upsample from 25hz to 30hz"
    #     "2020-capture24":   64,
    #     "2020-GOTOV":       64,     # "upsample from 88hz to 100hz"
    # }

    dataset_to_dev_mod = {
        # '2012-OPPORTUNITY': {   'Accelerometer+L':      ['acc'],
        #                         'Accelerometer+L01':    ['acc'],
        #                         'Accelerometer+R':      ['acc'],
        #                         'Accelerometer+R01':    ['acc']},
        # '2012-PAMAP2': {'IMU+L': ['acc', 'gyro', 'magnetic'],
        #                 'IMU+R': ['acc', 'gyro', 'magnetic'],
        #                 'IMU01+L': ['acc'],
        #                 'IMU01+R': ['acc']},

        # '2012-realdisp': {'IMU+L01':    ['acc', 'gyro', 'magnetic'],
        #                 'IMU+R01':      ['acc', 'gyro', 'magnetic']},

        '2014-mHealth': {'IMU+R':       ['acc', 'gyro', 'magnetic']},
        # '2014-shoaib': {'SMARTPHONE+R': ['acc', 'gyro', 'linearAcc', 'magnetic']},
        # '2015-HHAR': {'gear01+D':   ['acc', 'gyro'],
        #             'gear02+D':     ['acc', 'gyro'],
        #             'lgwatch01+D':  ['acc', 'gyro'],
        #             'lgwatch02+D':  ['acc', 'gyro']},
        # '2019-WISDM': {'watch+D':   ['acc', 'gyro']},
        # '2020-GOTOV': {'GENEActiv+R': ['acc']},
        # '2020-capture24': {'AX3+D': ['acc']}
    }

    dataset_to_mod = {
        d: reduce(lambda re, x: re + [x,] if x not in re else re, reduce(lambda re, x: re + x, [dataset_to_dev_mod[d][k] for k in dataset_to_dev_mod[d]], []), []) 
        for d in dataset_to_dev_mod
    }
    
    dataset_mod_params = []
    for d in dataset_to_mod:
        for m in dataset_to_mod[d]:
            dataset_mod_params += [(d, m)]

    CUR_FOLD = args_global.TAR_FOLD
    TEST_SAMPLE_TIME_WINDOW = args_global.TEST_TIME_WINDOW


    # prepare the parameters
    params_ = []
    for curDataset, curModality in dataset_mod_params:
        if curDataset not in dataset_to_window:
            continue
        # print(curDataset, curModality)

        # exit(0)
        if CUR_FOLD not in DATASET_TO_FOLDS[curDataset]:
            continue

        params_ += [(curDataset, curModality)]

    print(params_)

    
    for curDataset, curModality in params_:
        print(curDataset, curModality)

        MMAP_MODE = False


        curPlacement    = args_global.placement

        curTimeLen, curTimeWindow = timeWindowSetting[dataset_to_window[curDataset]]

        curTaskName     = DATASET_TO_TASK[curDataset][DEFAULT_GRAINED_TASK]
        curLabel        = grabTasklabels(curPlacement, curDataset, TEST_SAMPLE_TIME_WINDOW, curTaskName)

        EXP_NAME = str(args_global.EXP_NAME)
        if '_gru' in args_global.EXP_NAME:
            EXP_NAME = EXP_NAME.replace("_gru", "-gru")
        
        # if '_gru' in args_global.EXP_NAME:
        #     args_global.EXP_NAME = args_global.replace("_gru", "-gru")
        train_folder    = f"results{f'_w{TEST_SAMPLE_TIME_WINDOW}' if TEST_SAMPLE_TIME_WINDOW != 200 else ''}/{EXP_NAME}/{curPlacement}/{curDataset}/{curModality}/{CUR_FOLD}"

        model_path      = f"{train_folder}/model.pth"
        if not os.path.exists(model_path):
            print(f"Train: {train_folder} does not exist.", file=sys.stderr)
            continue
        
        conv_params = [
            (3, 96, 7, 1, 3),
            (96, 128, 5, 1, 2),
            (128, 144, 3, 1, 1)
        ]
        with open(f"results{f'_w{TEST_SAMPLE_TIME_WINDOW}' if TEST_SAMPLE_TIME_WINDOW != 200 else ''}/{EXP_NAME}/model_info.json", "r") as f:
            data = json.load(f)
            model_type = data["MODEL_TRAINING_STRATEGY"].split("-")[-2] if 'loaderLength' not in data["MODEL_TRAINING_STRATEGY"] else data["MODEL_TRAINING_STRATEGY"].split("-")[-3]
            if 'loaderLength' in data["MODEL_TRAINING_STRATEGY"] and "reg-v2wBERT" in data["MODEL_TRAINING_STRATEGY"]:
                model_type = data["MODEL_TRAINING_STRATEGY"].split("-")[-4]
            path2wpf = data["MODEL_FOLDER"]
            trainMechanism = data["MODEL_TRAINING_STRATEGY"]
            print(path2wpf, trainMechanism, data["MODEL_VERSION_IN_STEP"])

            if 'conv_params' in data:
                conv_params = data['conv_params']


            
        with open(f"{path2wpf}/{trainMechanism}/_metaInfo.json", "r") as f:
            cfg = AttrDict(json.load(f))

        print(cfg.keys())

        cfg.quantizer_cfg = AttrDict(cfg.quantizer_cfg)
        context_feature_encoder = WearablePretrainedEncoder(cfg.TEST_SAMPLE_TIME_WINDOW // 200, 
                                                            cfg.feature_size, 
                                                            cfg.hidden_dim,
                                                            use_quantizer           = True,
                                                            codebook_num_groups     = cfg.quantizer_cfg.codebook_num_groups,
                                                            codebook_num_vars       = cfg.quantizer_cfg.codebook_num_vars,
                                                            codebook_vq_dim         = cfg.quantizer_cfg.codebook_vq_dim,
                                                            quantizer_training_temp = cfg.quantizer_cfg.quantizer_training_temp,
                                                            use_trainable_mask      = cfg.use_trainable_mask,
                                                            share_layers_t1         = cfg.shared_layers,
                                                            share_layers_t2         = cfg.shared_layers,
                                                            version                 = model_type
                                                        )
        datasetKey      = f"{curDataset}"

        print(f": {datasetKey}/{curModality} :")

        # fold - identify the group of the test
        curFoldTestUsers   = DATASET_TO_FOLDS[curDataset][CUR_FOLD][1] # only use the group of training.

        # Optimizer
        # None:     lr set to 1e-3
        # _lr1e-4 : lr set to 1e-4 < seems learning rate do not have that big effect on the training results in 100 epochs.

        # Loss Function
        criterion   = nn.CrossEntropyLoss()

        test_data   = PretrainDataset(f"{DATA_CATEGORY_PATH}/{curPlacement}", curDataset, 
                                    transform_x=ResampleTransform(timeLength=curTimeLen, timeWindow=curTimeWindow, method="nearest"), 
                                    trainValSplit='all', 
                                    taskName=curTaskName, 
                                    sampleTimeWindow=TEST_SAMPLE_TIME_WINDOW, 
                                    sampleMiniWindow=True, # if True it means it will split in pieces.
                                    verbose=0,
                                    targetFolders=TarFolderFilter(modality=[curModality,], user=curFoldTestUsers),
                                    MMAP_MODE = MMAP_MODE,
                                    EXCLUDE_NONE = True,
                                    HDF5=True
                                )
        test_loader = DataLoader(test_data, batch_size=128, pin_memory=True, num_workers=4)
        
        if len(conv_params) != 0:
            model       = grab_avgpool_classifier(conv_params, input_size=curTimeLen, output_size=len(curLabel) - 1)
        else:
            numChannels = 3
            model       = grab_avgpool_classifier_linear([[numChannels, ]], input_size=curTimeLen, output_size=len(curLabel) - 1)
            
        # full-retraining setting
        model.feature = FeatureWrapper(dataset_feature_extractor=model.feature, context_feature_extractor=context_feature_encoder, use_drop_out=False)
        
        # Find optimal batch size
        device      = get_device()
        

        test_model(model, model_path, test_loader, device, log_dir=f'{train_folder}/test-logs')
