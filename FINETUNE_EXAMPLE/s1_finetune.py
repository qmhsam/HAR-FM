
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import reduce
from models import  WearablePretrainedEncoder, FeatureWrapper, \
                    grab_avgpool_classifier, grab_avgpool_classifier_linear
from utils import *
from tqdm import tqdm

import csv, json
from datetime import datetime

from collections import deque
import threading

import sys, os

import argparse 
parser = argparse.ArgumentParser(description='tbc')
parser.add_argument('--TAR_FOLD', default=-1, type=int, help='k-fold setting', choices=[0, 1, 2, 3, 4]) # -1: train with all users

parser.add_argument('--EXP_NAME', default="test", type=str)
parser.add_argument('--TEST_TIME_WINDOW', default=800, type=int, choices=[800, 1000, 2000, 3000, 5000, 6000])

# 
parser.add_argument("--DENSE_SAMPLE", type=int, default=15, choices=[0, 15])
parser.add_argument('--POST_MODEL_FOLDER', default="EP1_RESULTS/2_HAR-FM/exp-full-dropout-2048-real-pfm34-MSE-mpnet-ext-loaderLength-v1+rand_view", type=str)
parser.add_argument('--POST_MODEL_VERSION_IN_STEP', default=50000, type=int)

parser.add_argument("--TrainFromScratch", default=False, action="store_true")

parser.add_argument("--encoder_layer_num", type=int, choices=[0, 2, 3], default=0)

args_global = parser.parse_args()

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # return torch.device('cpu')

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DataPrefetcherThread:
    def __init__(self, loader, device, num_prefetched_batches=300):
        self.loaderit = None
        self.loader = loader
        self.device = device
        self.num_prefetched_batches = num_prefetched_batches
        self.prefetched_batches = deque()
        self.loader_lock = threading.Lock()
        self.stream = torch.cuda.Stream()
        self.running = True

    def preload_batches(self):
        while self.running:
            if len(self.prefetched_batches) < self.num_prefetched_batches:
                with self.loader_lock:
                    try:
                        next_input, next_target, next_metaInfo = next(self.loaderit)
                    except StopIteration:
                        self.running = False
                        break  # No more data to load

                with torch.cuda.stream(self.stream):
                    next_input = next_input.cuda(device=self.device, non_blocking=False)
                    next_target = next_target.cuda(device=self.device, non_blocking=False)
                    next_metaInfo = {k: v.cuda(device=self.device, non_blocking=False) if isinstance(v, torch.Tensor) else v
                                     for k, v in next_metaInfo.items()}
                    
                    # Append the new batch to the deque
                    self.prefetched_batches.append((next_input, next_target, next_metaInfo))
            else:
                # Sleep a bit to prevent this loop from hogging the CPU
                threading.Event().wait(0.01)

    def start_prefetching(self):
        self.prefetch_thread = threading.Thread(target=self.preload_batches)
        self.prefetch_thread.start()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)

        while len(self.prefetched_batches) == 0 and self.running:
            # Sleep a bit to wait for batches to be prefetched
            print("[INFO] loading batches from the memory.", file=sys.stderr)
            threading.Event().wait(0.01)

        if len(self.prefetched_batches) == 0 and not self.running:
            return None

        return self.prefetched_batches.popleft()

    def __iter__(self):
        if self.loaderit is None:
            self.loaderit = iter(self.loader)

        self.start_prefetching()
        while True:
            batch = self.next()
            if batch is None:
                self.loaderit = None
                break
            yield batch

    def __len__(self):
        return len(self.loader)

    def __del__(self):
        self.running = False
        if self.prefetch_thread.is_alive():
            self.prefetch_thread.join()

# Training Loop
def train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, num_epochs, device, patience=100, 
                save_path='best_model.pth', 
                log_dir="logs"):
    
    model.to(device)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_log_{timestamp}.csv')

    # Open the log file and write the header
    with open(log_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Accuracy', 'Validation Loss', 'Learning Rate'])

        
    for epoch in range(num_epochs):
        model.train()

        if train_loader.dataset.MMAP_MODE:
            train_loader.dataset.shuffleInFile() # shuffle each epoch
        if epoch > 0 and args_global.DENSE_SAMPLE != 0:
            train_loader.dataset.prepare_sample_offset(init=False, segmentation=args_global.DENSE_SAMPLE) # add offset to resample the signal.

        train_prefetcher    = DataPrefetcherThread(train_loader, device)
        train_loop          = tqdm(train_prefetcher, desc=f'Epoch {epoch+1}/{num_epochs} [Training]', leave=False, mininterval=5)

        total_loss = 0.
        # permute_ = torch.randperm(3)
        for data, target, _ in train_loop:
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target-1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(train_loader)
        # Validation
        model.eval()
        test_prefetcher = DataPrefetcherThread(test_loader, device)
        val_loop = tqdm(test_prefetcher, desc=f'Epoch {epoch+1}/{num_epochs} [Validation]', leave=False, mininterval=5)
        
        val_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Evaluate on the validation set
        with torch.no_grad():
            val_criterion = nn.CrossEntropyLoss()

            for data, target, _ in val_loop:
                # data = data.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == target - 1).sum().item()
                total_samples += target.size(0)
                val_loss += val_criterion(output, target - 1).item()

        val_loss /= len(val_loop)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        accuracy = total_correct / total_samples

        # Log the metrics
        with open(log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, total_loss, accuracy, val_loss, current_lr])

        print(f'Epoch {epoch+1}, Loss: {total_loss: 7.5f}, Accuracy: {accuracy: 7.5f}, Eval Loss: {val_loss: 7.5f}')
        
        # TODO: 
        # 1- early stop.
        if val_loss < best_val_loss:
            print(f'New best model found at epoch {epoch+1} with Validation Loss: {val_loss: 7.5f}')
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the model
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path}')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping after {epoch+1} epochs')
                break

        # Step the scheduler
        if scheduler is not None:
            scheduler.step(total_loss)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    
    post_train_folder       = args_global.POST_MODEL_FOLDER
    
    timeWindowSetting = {
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

    

    dataset_to_dev_mod = {
        # '2012-PAMAP2': {'IMU+L': ['acc', 'gyro', 'magnetic'],
        #                 'IMU+R': ['acc', 'gyro', 'magnetic'],
        #                 'IMU01+L': ['acc'],
        #                 'IMU01+R': ['acc']},

        '2014-mHealth': {'IMU+R':       ['acc', 'gyro', 'magnetic']},

        # '2014-shoaib': {'SMARTPHONE+R': ['acc', 'gyro', 'linearAcc', 'magnetic']},

        # '2015-HHAR': {'gear01+D':   ['acc', 'gyro'],
        #             'gear02+D':     ['acc', 'gyro'],
        #             'lgwatch01+D':  ['acc', 'gyro'],
        #             'lgwatch02+D':  ['acc', 'gyro']},

        # '2020-GOTOV': {'GENEActiv+R': ['acc']},
    }

    dataset_to_mod = {
        d: reduce(lambda re, x: re + [x,] if x not in re else re, reduce(lambda re, x: re + x, [dataset_to_dev_mod[d][k] for k in dataset_to_dev_mod[d]], []), []) 
        for d in dataset_to_dev_mod
    }
    
    dataset_mod_params = []

    for d in dataset_to_mod:
        for m in dataset_to_mod[d]:
            dataset_mod_params += [(d, m)]

    CUR_FOLD = max(args_global.TAR_FOLD, 0) # -1 will use the parameters of the 0 fold.
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
    
    # Find optimal batch size
    device = get_device()
    # 
    with open(f"{post_train_folder}/_metaInfo.json", "r") as f:
        cfg = AttrDict(json.load(f))

    model_type =        cfg.pretrained_cfg["model_type"]
    path2wpf =          cfg.pretrained_cfg["path2wpf"]
    trainMechanism =    cfg.pretrained_cfg["trainMechanism"]
    
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
    
    trainStep =         args_global.POST_MODEL_VERSION_IN_STEP
    dataDict= AttrDict(torch.load(f"{post_train_folder}/history/model_checkpoint_step_{trainStep}.pth", map_location=device))
    
    for curDataset, curModality in params_:
        print(curDataset, curModality)
        
        MMAP_MODE = False

        curPlacement    = "WRIST"

        curTimeLen, curTimeWindow = timeWindowSetting[dataset_to_window[curDataset]]
        # curTimeLen      = int(curTimeLen * (TEST_SAMPLE_TIME_WINDOW // 200))
        # curTimeWindow   = int(curTimeWindow * (TEST_SAMPLE_TIME_WINDOW // 200))

        curTaskName     = DATASET_TO_TASK[curDataset][DEFAULT_GRAINED_TASK]
        curLabel        = grabTasklabels(curPlacement, curDataset, TEST_SAMPLE_TIME_WINDOW, curTaskName)

        # curModality         = "acc"

        # fold - identify the group of the test
        curFoldTrainUsers = None
        if args_global.TAR_FOLD != -1:
            curFoldTrainUsers   = DATASET_TO_FOLDS[curDataset][CUR_FOLD][0] # only use the group of training.
        # fewer encoder parameters
        encode_postfix  = ''
        if args_global.encoder_layer_num == 0:
            encode_postfix = f"_linear"
        elif args_global.encoder_layer_num != 3:
            encode_postfix = f"_c{args_global.encoder_layer_num}"

        parent_folder   = f"./results{f'_TrainFromScratch' if args_global.TrainFromScratch else ''}{f'_w{TEST_SAMPLE_TIME_WINDOW}' if TEST_SAMPLE_TIME_WINDOW != 200 else ''}/{args_global.EXP_NAME}{encode_postfix}_finetuneAll"
        train_folder    = f"{parent_folder}/{curPlacement}/{curDataset}/{curModality}/{args_global.TAR_FOLD if args_global.TAR_FOLD != -1 else 'all'}" # specific models.

        if os.path.exists(f"{train_folder}/model.pth"):
            continue
        
        datasetKey = f"{curDataset}"

        print(f": {datasetKey}/{curModality} :")


        conv_params = [
            [3, 96, 7, 1, 3],
            [96, 128, 5, 1, 2],
            [128, 144, 3, 1, 1]
        ]
        conv_params = conv_params[:args_global.encoder_layer_num]
        print(conv_params)

        if args_global.encoder_layer_num != 0:
            model       = grab_avgpool_classifier(conv_params, input_size=curTimeLen, output_size=len(curLabel) - 1, use_drop_out=False)
        else:
            model       = grab_avgpool_classifier_linear([[3, ]], input_size=curTimeLen, output_size=len(curLabel) - 1, use_drop_out=False)

        if not args_global.TrainFromScratch:
            context_feature_encoder.load_state_dict(dataDict.model_state_dict)


        # full-retraining setting
        model.feature = FeatureWrapper(dataset_feature_extractor=model.feature, context_feature_extractor=context_feature_encoder, use_drop_out=False) # -2 : True
        # Optional
        model.feature.contextFeature.freeze_quantizer() 

        optimizer       = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-6, amsgrad=True)


        train_data = PretrainDataset(f"{DATA_CATEGORY_PATH}/{curPlacement}", curDataset, 
                                    transform_x=ResampleTransform(timeLength=curTimeLen, timeWindow=curTimeWindow, method="nearest"), 
                                    trainValSplit=None, 
                                    taskName=curTaskName, 
                                    sampleTimeWindow=TEST_SAMPLE_TIME_WINDOW, 
                                    sampleMiniWindow=True, # if True it means it will split in pieces.
                                    verbose=0,
                                    targetFolders=TarFolderFilter(modality=[curModality,], user=curFoldTrainUsers),
                                    EXCLUDE_NONE = True,
                                    MMAP_MODE = MMAP_MODE, 
                                    SHUFFLE   = MMAP_MODE,
                                    HDF5=True,
                                    POST_SPLIT=0, # -1 (default), 0 - train, 1 - test
                                )
        
        # Loss Function
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_data, batch_size=64, pin_memory=True, num_workers=4, shuffle=not MMAP_MODE)
        test_data = PretrainDataset(f"{DATA_CATEGORY_PATH}/{curPlacement}", curDataset, 
                                    transform_x=ResampleTransform(timeLength=curTimeLen, timeWindow=curTimeWindow, method="nearest"), 
                                    trainValSplit=None, 
                                    taskName=curTaskName, 
                                    sampleTimeWindow=TEST_SAMPLE_TIME_WINDOW, 
                                    sampleMiniWindow=True, # if True it means it will split in pieces.
                                    verbose=0,
                                    targetFolders=TarFolderFilter(modality=[curModality,], user=curFoldTrainUsers),
                                    EXCLUDE_NONE = True,
                                    MMAP_MODE = MMAP_MODE,
                                    HDF5 = True,
                                    POST_SPLIT=1
                                )
                                


        test_loader = DataLoader(test_data, batch_size=128, pin_memory=True, num_workers=4)

        os.makedirs(parent_folder, exist_ok=True)
        with open(f"{parent_folder}/model_info.json", "w") as f:
            json.dump({
                "MODEL_FOLDER":                 path2wpf,
                "MODEL_TRAINING_STRATEGY":      trainMechanism,
                "MODEL_VERSION_IN_STEP":        trainStep,
                "POST_MODEL_FOLDER":            post_train_folder,
                "conv_params": conv_params
            }, f, indent=4)
        
        
        nb_epochs = 1000
        # Call the training function
        train_model(model, train_loader, test_loader, optimizer, criterion, None, nb_epochs, device=device, 
                    save_path=f"{train_folder}/model.pth", 
                    log_dir=f"{train_folder}/logs")
