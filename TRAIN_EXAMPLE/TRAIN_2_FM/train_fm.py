# 
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import traceback

from collections import Counter
import logging
import json
from utils import *
from models import grabClassifier, AutoEncoder, WearablePretrainedEncoder, FwBlock, LinearFeatureExtractor2, ClassAvgPoolProjector

from torch.utils.tensorboard import SummaryWriter
# torch.autograd.set_detect_anomaly(True)

import argparse
# Changed from:     train_1_pretrainedModel-contrastiveLossTopk_bs1024_codebook_amsgrad_UP-step0_lr_tau_randFeat.py
#   README:         change the contrastive learning loss computation to top k elements with a stride of 3 << 600 ms
# projector only instead of customized cnn or what
parser = argparse.ArgumentParser(description='pretraining foundation model')
parser.add_argument('--TRAIN_SOURCE', default=False, action="store_true", help='if train the low-level dataset-wise encoder')

parser.add_argument('--DATASET_SEGMENTATION', default=15, type=int, choices=[15, 25, 150])   

parser.add_argument('--MODEL_STRUCTURE', default="pfm34", type=str)
# 
parser.add_argument('--MODEL_FOLDER', default="../TRAIN_1_SSL/results/g6v4-amsgrad-topK-UD-step0-lr2", type=str)
parser.add_argument('--MODEL_TRAINING_STRATEGY', default="exp-coarse-0-pfm34-v2wBERT-loaderLength", type=str)
parser.add_argument('--MODEL_VERSION_IN_STEP', default=1, type=int)

parser.add_argument('--EMBEDDING_MODEL', default="all-mpnet-base-v2", type=str, choices=["all-mpnet-base-v2", "all-MiniLM-L12-v2", "all-distilroberta-v1"])
parser.add_argument("--UA_LOSS_TYPE", default=0, type=int, choices=[0, ]) # 0 for MSE, 1 for SoftDTW
# optim: only work for MSE


parser.add_argument('--loss_style', default=0, type=int, choices=[0, 1, 2, 3, 4])
# DESC. Ablation study
# 0: all used
# 1: without task loss
# 2: without ua_loss
# 3: without mlm_loss
# 4: without pretrained

# parser.add_argument('--LOSS_TYPE', default=0, type=int, choices=[0, 1, 2, 3])   

# DESC.
# @LOSS_TYPE:   contrastive  |  masked language model (CE) | masked language model (MSE)    |   Similar Work            |
#   0       :   o            |  o                          | x                              |   v2wBERT                 |
#   1       :   o            |  x                          | o                              |   v2wBERT+regression MLM  |
#   2       :   x            |  x                          | o                              |   BERT                    |
#   3       :   o            |  x                          | x                              |   wav2vec2.0              |

args_global = parser.parse_args()

EMBEDDING_to_SHAPE = {
    "all-mpnet-base-v2": 768,
    "all-distilroberta-v1": 768,
    "all-MiniLM-L12-v2": 384
}

EMBEDDING_MODEL_SHORTCUT = {
    "all-mpnet-base-v2": 'mpnet',
    "all-distilroberta-v1": 'distil-roberta',
    "all-MiniLM-L12-v2": 'MiniLM-L12'
}

def get_device():
    # if torch.cuda.is_available():
    #     print(f"CUDA version: {torch.version.cuda}")
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


timeWindowSetting = {
    "30hz":     (6, 200),
    "50hz":     (10, 200),
    "100hz":    (20, 200)
} 

device = get_device()

def compute_post_train_loss(metaInfo, target, # batch info
                            info_, 
                            aux_output, 
                            task_output, 
                            ua_output,
                            mlm_criterion = nn.CrossEntropyLoss(),  
                            task_criterion = nn.CrossEntropyLoss(), 
                            ua_criterion = nn.MSELoss(),
                            return_mlm_task_results = False
                            ):
    
    # mlm loss
    # aux_feat_ = auxModel(feat_) if auxModel is not None else None
    # aux_feat_ = aux_feat_.reshape(bsz * tsz, -1)

    mlm_loss = torch.tensor(0.0).float()
    correct_mlm = 0
    if aux_output is not None:
        target_ids  = info_["quantized_ids"]
        b, t ,g = target_ids.shape
        target_ids  = target_ids.reshape(-1, target_ids.shape[-1])
        split_len   = aux_output.shape[-1] // target_ids.shape[-1]

        mlm_loss    = torch.stack([mlm_criterion(aux_output[:, i * split_len: (i + 1) * split_len], 
                                                 target_ids[:, i]) 
                                    for i in range(target_ids.shape[-1])]).mean()
        
        pred_ = aux_output.view(-1, target_ids.shape[-1], split_len).argmax(-1)
        correct_mlm = (pred_ == target_ids).all(-1).sum().float().item() / t
 
    # task_output = taskModel_(feat_.reshape(bsz, -1))
    task_loss = torch.tensor(0.0).float()
    if task_output is not None:
        task_loss   = task_criterion(task_output, target - 1)

    # alignment loss
    ua_loss = torch.tensor(0.0).float()
    if ua_output is not None:
        ua_loss     = ua_criterion(ua_output, metaInfo["embeddingSeq"].float()).mean().float()

    # grab loss sum lambda = 0.1, 
    if args_global.loss_style == 0:
        loss_sum    = mlm_loss + task_loss  + ua_loss
    if args_global.loss_style == 1:
        loss_sum    = mlm_loss              + ua_loss
    if args_global.loss_style == 2:
        loss_sum    = mlm_loss + task_loss
    if args_global.loss_style == 3 or args_global.loss_style == 4:
        loss_sum    =            task_loss  + ua_loss


    if return_mlm_task_results:
        return loss_sum, mlm_loss, task_loss, ua_loss, correct_mlm
    return loss_sum, mlm_loss, task_loss, ua_loss

    
def post_train_model(models, 
                     dataLoaderList, 
                     dataset_modality_to_encoder,
                     dataset_task_to_classifier, 
                    # task classification loss for each modality
                    # global alignment loss <- averaged in averaged mse loss from calls.
                    # each iteration calls 1 to 2 times
                    optimizer, 
                    scheduler, 
                    batch_size,
                    num_steps, 
                    universal_alignment_loss_style,  # 0-MSE or 1-SoftDTW
                    device, 
                    save_path="./results/undefined", 
                    log_path="./results/undefined/log",
                    writer = None,
                    maximum_taskNum = 205):
    assert len(models) in [1, 2, 3]
    tarModel = models[0]
    auxModel = models[1] if len(models) > 1 else None
    universal_alignment_aux_model = models[2] if len(models) > 2 else None

    # criterion definition
    ua_criterion = None
    if universal_alignment_loss_style == 0: #MSE
        ua_criterion = nn.MSELoss(reduction='none')
    assert universal_alignment_loss_style == 0, 'SoftDTW loss is not implemented.'
    # elif universal_alignment_loss_style == 1: # softDTW
    #     ua_criterion = SoftDTW(torch.cuda.is_available(), gamma=1.0, normalize=True)

    mlm_criterion = nn.CrossEntropyLoss(reduction='none')

    task_criterion = nn.NLLLoss(reduction='none')

    
    mlm_criterion_val = nn.CrossEntropyLoss()  
    task_criterion_val = nn.NLLLoss() 
    ua_criterion_val = nn.MSELoss()
    # 
    # Set up logging and tensorboard
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_path, 'training.log'), level=logging.INFO)
    
    TASK_NUM = sum([k[-2] for k in dataLoaderList])
    for step in range(num_steps):
        tarModel.train()
        if auxModel is not None:
            auxModel.train()
        if universal_alignment_aux_model is not None:
            universal_alignment_aux_model.train()
        
        train_loss_sum = torch.tensor(0.0, device=device).float()
        task_loss_sum = 0.0
        loss_component = {
            "mlm_loss": 0.0,
            "ua_loss" : 0.0
        }
        randperm_ids = torch.randperm(96)
        extra_optimizers = []
        used_encoder_key  = []
        num_elements = 10
        datasetTaskDataStack = {}
        for d, m, taskName, train_dLoader, _, taskWeight, dataVolume in tqdm(dataLoaderList, "[train] extracting dataset-wise feature."):
            dataset_taskName_key = (d, taskName)
            if dataset_taskName_key not in datasetTaskDataStack:
                datasetTaskDataStack[dataset_taskName_key] = {
                    "dataset_feature": [],
                    "embeddingSeq": [],
                    "target": [],
                    "dataVolume": 0
                } # (datasetwise feat, embedded sequence, target label)

            # print(d, m, file=sys.stderr)
            batch = next(train_dLoader)
            data, target, metaInfo = batch
            data, target = data.to(device), target.to(device)
            metaInfo["embeddingSeq"] = metaInfo["embeddingSeq"].to(device)
            bsz, tsz, csz, fsz = data.shape

            # mask = torch.zeros(tsz, dtype=torch.bool)
            # indices = torch.randperm(tsz)[:num_elements]
            # mask[indices] = True

            data = data.reshape(-1, csz, fsz)
            
            encoder, opt = dataset_modality_to_encoder[(d, m)]
            if (d, m) not in used_encoder_key:
                used_encoder_key += [(d, m)]

            if args_global.TRAIN_SOURCE:
                encoder.train()
                extra_optimizers += [opt,]
            else:
                encoder.eval()
                
            dataset_wise_feat = encoder(data)
            dataset_wise_feat = dataset_wise_feat.view(bsz, tsz, -1)

            DATASET_FEAT_DROPOUT_WEIGHT_1 = F.dropout(torch.ones(dataset_wise_feat.shape[-1], device=device), 0.3)
            DATASET_FEAT_DROPOUT_WEIGHT_2 = F.dropout(torch.ones(dataset_wise_feat.shape[-1], device=device), 0.3)
            
            dataset_wise_feat_cat = torch.concat([F.layer_norm(dataset_wise_feat * DATASET_FEAT_DROPOUT_WEIGHT_1, (dataset_wise_feat.shape[-1], )), 
                                                  F.layer_norm(dataset_wise_feat * DATASET_FEAT_DROPOUT_WEIGHT_2, (dataset_wise_feat.shape[-1], ))]) # [2*bsz, tsz, fsz]
            # info_ = {"embeddingSeq": metaInfo["embeddingSeq"].float().repeat(2, 1, 1)}, target = target.repeat(2),
            datasetTaskDataStack[dataset_taskName_key]["dataset_feature"] += [dataset_wise_feat_cat,]
            datasetTaskDataStack[dataset_taskName_key]["embeddingSeq"]    += [metaInfo["embeddingSeq"].float().repeat(2, 1, 1),]
            datasetTaskDataStack[dataset_taskName_key]["target"]          += [target.repeat(2),]
            datasetTaskDataStack[dataset_taskName_key]["dataVolume"]      += dataVolume
#           
        
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()

        contextual_feature = []
        stacked_embeddingSeq = []
        stacked_quantized_ids = []
        loss_weight  = []

        stacked_tasks_target = []
        stacked_tasks_output = []
        stacked_tasks_weights = []

        DENORM = sum([datasetTaskDataStack[d]["dataVolume"] for d in datasetTaskDataStack])
        for d, taskName in tqdm(datasetTaskDataStack, "[train] computing the loss"):
            
            # register the taskName
            dataset_taskName_key = (d, taskName)
            # if '+'.join(dataset_taskName_key) not in loss_component:
            #     loss_component['+'.join(dataset_taskName_key)] = 0.0

            taskModel_ = dataset_task_to_classifier[dataset_taskName_key]
            
            if args_global.loss_style != 1:
                taskModel_.train()
            else:
                taskModel_.eval()

            dataset_wise_feat = torch.concat(datasetTaskDataStack[(d, taskName)]["dataset_feature"])
            
            feat_cat, info_cat = tarModel(dataset_wise_feat) # 
            
            contextual_feature += [feat_cat.reshape(-1, feat_cat.size(-1)),]

            cur_task_dataset_weight = (1.0 * datasetTaskDataStack[(d, taskName)]["dataVolume"]) / DENORM
            print(d, taskName, cur_task_dataset_weight, file=sys.stderr)
            loss_weight += [(torch.zeros(feat_cat.size(0) * feat_cat.size(1), device=device) + 1.0 / feat_cat.size(0) / feat_cat.size(1) * cur_task_dataset_weight).float()]

            stacked_embeddingSeq += [torch.concat(datasetTaskDataStack[(d, taskName)]["embeddingSeq"])
                                     .reshape(-1, EMBEDDING_to_SHAPE[args_global.EMBEDDING_MODEL])] 
            # NOTICE: It will raise problem if it goes to softdtw, but not MSE

            stacked_quantized_ids += [info_cat["quantized_ids"].reshape(-1, tarModel.codebook_num_groups), ]
            
            if args_global.loss_style != 1:
                task_output_cat = taskModel_(feat_cat)

                # padding length 
                pad_len = maximum_taskNum - task_output_cat.size(-1)
                stacked_tasks_output += [torch.concat([task_output_cat, torch.zeros(1, pad_len).expand(task_output_cat.size(0), pad_len).to(device=device)], 1), ]
                stacked_tasks_target += [torch.concat(datasetTaskDataStack[(d, taskName)]["target"]) - 1]
                stacked_tasks_weights += [torch.zeros(task_output_cat.size(0)).to(device=device) + 1.0 / task_output_cat.size(0) * cur_task_dataset_weight]

            # task_loss = task_criterion(task_output_cat, torch.concat(datasetTaskDataStack[(d, taskName)]["target"]) - 1)

            # task_loss_sum                     += task_loss
            # loss_component['+'.join(dataset_taskName_key)]  += task_loss.item() 
        
        # Start timing
        start_time = time.time()

        del datasetTaskDataStack
        # torch.cuda.empty_cache()
        task_loss_sum = torch.tensor(-1e-6, device=device).float()
        if args_global.loss_style != 1:
            task_loss_sum = (task_criterion(torch.concat(stacked_tasks_output), torch.concat(stacked_tasks_target)) * torch.concat(stacked_tasks_weights)).sum()


        # compute the mlm_loss and ua_loss together
        feat_cat = torch.concat(contextual_feature)
        weight2sample = torch.concat(loss_weight)
        del loss_weight
        del contextual_feature
        del stacked_tasks_output
        del stacked_tasks_target
        del stacked_tasks_weights
        # torch.cuda.empty_cache()

        # compute mlm_loss
        aux_output = auxModel(feat_cat) if auxModel is not None else None
        target_ids  = torch.concat(stacked_quantized_ids)
        split_len   = aux_output.shape[-1] // target_ids.shape[-1]
        mlm_loss    = (torch.stack([mlm_criterion(aux_output[:, i * split_len: (i + 1) * split_len], 
                                                 target_ids[:, i]) 
                                    for i in range(target_ids.shape[-1])]) * weight2sample.unsqueeze(0)).mean(0).sum()
        
        del stacked_quantized_ids
        # torch.cuda.empty_cache()
        
        # compute the ua_loss
        ua_output  = universal_alignment_aux_model(feat_cat)
        target_embeddingSeq = torch.concat(stacked_embeddingSeq)
        # ONLY work for MSE
        ua_loss = (ua_criterion(ua_output, target_embeddingSeq.float()).mean(1) * weight2sample).sum()
        
        del stacked_embeddingSeq
        # torch.cuda.empty_cache()

        # loss_style control
        if args_global.loss_style in [0, 1]:
            train_loss_sum += (mlm_loss + ua_loss)

        if args_global.loss_style == 2:
            train_loss_sum += mlm_loss

        if args_global.loss_style in [3, 4]: 
            train_loss_sum += ua_loss

        loss_component["ua_loss"]           += ua_loss.item() 
        loss_component["mlm_loss"]          += mlm_loss.item()

        # task_loss_sum         /= 
        
        # log each step
        writer.add_scalar('Train/loss_sum',  train_loss_sum.item(), step)
        for name, value in loss_component.items():
            writer.add_scalar(f'Train/{name}', value, step)

        # backward step by step
        #  for train_loss_sum, it updates the context encoder and aux models
        #  for task_loss, it only updates the task classifiers

        
        # stage 1, fixed classifier, update backbone and aux Models
        # zero_grad
        optimizer.zero_grad()
        for opt in extra_optimizers:
            opt.zero_grad()
        
        if args_global.loss_style != 1:
            for k in used_encoder_key:
                # encoder, _ = dataset_modality_to_encoder[k]
                for params in dataset_modality_to_encoder[k][0].parameters():
                    params.requires_grad = False

        # backward
        train_loss_sum.backward(retain_graph=True if args_global.loss_style != 1 else False)

        # stage 2:
        
        if args_global.loss_style != 1:
            # unfreeze the encoder
            for k in used_encoder_key:
                # encoder, _ = dataset_modality_to_encoder[k]
                for params in dataset_modality_to_encoder[k][0].parameters():
                    params.requires_grad = True
        
        if args_global.loss_style != 1:
            task_loss_sum.backward()
        
        optimizer.step()
        for opt in extra_optimizers:
            opt.step()

        if scheduler is not None:
            scheduler.step()
        
        # End timing
        end_time = time.time()
        runtime = end_time - start_time

        # Print runtime to sys.stderr
        print(f"Loss backward Runtime: {runtime:.6f} seconds", file=sys.stderr)

        
        optimizer.zero_grad()
        for opt in extra_optimizers:
            opt.zero_grad()        

        task_loss_sum  = task_loss_sum.item()
        train_loss_sum = train_loss_sum.item()
        torch.cuda.empty_cache()
        # torch.cuda.synchronize()

        if (step + 1) % 2500 == 0 or step == 0:
            tarModel.eval()
            if auxModel is not None:
                auxModel.eval()
            if universal_alignment_aux_model is not None:
                universal_alignment_aux_model.eval()

            # TODO: validation computation            
            val_loss_sum = 0.0

            val_loss_component = {
                "ua_loss":  0.0,
                "mlm_loss": 0.0
            }

            val_task_accuracy = {}

            for d, m, taskName, _, val_dLoader, taskWeight, _ in dataLoaderList:
                dm_total_size= len(val_dLoader.dataset)
                print(d, m, f": {dm_total_size}", file=sys.stderr)
                dm_val_loss_sum, dm_l_mlm, dm_l_task, dm_l_ua  = 0., 0., 0., 0.
                
                dataset_taskName_key = (d, taskName)
                if ('task_loss/' + '+'.join(dataset_taskName_key)) not in val_loss_component:
                    val_loss_component['task_loss/' + '+'.join(dataset_taskName_key)] = 0.0
                
                dataset_modality_task_key = (d, m, taskName)

                total_correct = 0
                total_correct_mlm = 0

                taskModel_ = dataset_task_to_classifier[dataset_taskName_key]
                taskModel_.eval()
                if args_global.loss_style != 1:
                    dLoader = tqdm(DataPrefetcherThread(val_dLoader, device=device), desc=f'Step {step+1}/{num_steps} [Valid on {d:>12s}_{m:9s}]', leave=False, mininterval=5)
                else:
                    dLoader = tqdm(val_dLoader, desc=f'Step {step+1}/{num_steps} [Valid on {d:>12s}_{m:9s}]', leave=False, mininterval=5)

                for batch in dLoader:
                    data, target, metaInfo = batch
                    
                    if args_global.loss_style == 1:
                        data = data.to(device)
                        target = target.to(device)
                        metaInfo["embeddingSeq"] = metaInfo["embeddingSeq"].to(device)

                    bsz, tsz, csz, fsz = data.shape

                    data = data.reshape(-1, csz, fsz)
                    
                    encoder, _ = dataset_modality_to_encoder[(d, m)]
                    encoder.eval()

                    with torch.no_grad():
                        dataset_wise_feat = encoder(data)
                        dataset_wise_feat = dataset_wise_feat.reshape(bsz, tsz, -1)
                        
                        feat_, info_ = tarModel(dataset_wise_feat)
                        aux_output  = auxModel(feat_).reshape(bsz * tsz, -1) if auxModel is not None else None
                        task_output = taskModel_(feat_)
                        _, task_pred = torch.max(task_output.data, 1)
                        total_correct += (task_pred == target - 1).sum().item()
                        ua_output   = universal_alignment_aux_model(feat_)

                        loss_sum, mlm_loss, task_loss, ua_loss, correct_mlm = compute_post_train_loss(
                                                metaInfo, target, # batch info
                                                info_, 
                                                aux_output, 
                                                task_output, 
                                                ua_output,
                                                mlm_criterion = mlm_criterion_val,  
                                                task_criterion = task_criterion_val, 
                                                ua_criterion = ua_criterion_val,
                                                return_mlm_task_results=True
                                                )
                        total_correct_mlm += correct_mlm
                        # train_loss_sum += loss_sum / 2 * taskWeight
                        # loss_component["mlm_loss"]                      += mlm_loss.item()  / 2 * taskWeight
                        # loss_component["ua_loss"]                       += ua_loss.item()   / 2 * taskWeight
                        # loss_component['+'.join(dataset_taskName_key)]  += task_loss.item() / 2 * taskWeight
                        # del aux_output
                        del task_output
                        # del ua_output
                    dm_val_loss_sum += loss_sum.item() * bsz
                    # 
                    dm_l_mlm    += mlm_loss.item()  * bsz
                    dm_l_task   += task_loss.item() * bsz
                    dm_l_ua     += ua_loss.item()   * bsz
                
                dm_val_loss_sum /= dm_total_size
                # 
                dm_l_mlm        /= dm_total_size
                dm_l_task       /= dm_total_size
                dm_l_ua         /= dm_total_size
                
                # log the validation for each dataset.
                loss_component_local = {
                    "task_loss":        dm_l_task,
                    "ua_loss" :         dm_l_ua,
                    "mlm_loss":         dm_l_mlm
                }
                writer.add_scalar(f"Valid/loss_sum-{'+'.join(dataset_taskName_key)}/{m}",  dm_val_loss_sum, step)
                for name, value in loss_component_local.items():
                    writer.add_scalar(f"Valid/{name}/{'+'.join(dataset_taskName_key)}/{m}", value, step)

                # update the summary
                val_loss_sum                                        += dm_val_loss_sum * taskWeight
                val_loss_component['task_loss/' + '+'.join(dataset_taskName_key)]  += dm_l_task * taskWeight
                val_loss_component["ua_loss"]                       += dm_l_ua * taskWeight
                val_loss_component["mlm_loss"]                      += dm_l_mlm * taskWeight

                print(f"{d:>16s}, {m:>8s}, {taskName:<20s}, mlm_acc = {total_correct_mlm / dm_total_size:.4f}, task_acc = {total_correct / dm_total_size:.4f}")

                val_task_accuracy['+'.join(dataset_modality_task_key)] = {
                    'mlm_acc':   total_correct_mlm / dm_total_size, 
                    'task_acc' : total_correct / dm_total_size
                }
            mean_mlm_acc = np.mean([v['mlm_acc'] for _, v in val_task_accuracy.items()])
            mean_task_acc = np.mean([v['task_acc'] for _, v in val_task_accuracy.items()])
            print(f"Average, mlm_acc = {mean_mlm_acc:.4f}, task_acc = {mean_task_acc:.4f}")

            denom   =   TASK_NUM
            val_loss_sum                                /= denom # reweight 
            # Task does not require denormalization since it has already done.
            val_loss_component["ua_loss"]               /= denom 
            val_loss_component["mlm_loss"]              /= denom
            
            writer.add_scalar(f'Valid/loss_sum/Avg',   val_loss_sum, step)
            for name, value in val_loss_component.items():
                writer.add_scalar(f'Valid/{name}/Avg',  value, step)

            log_message = f"Step {step + 1}, Train Loss: {train_loss_sum:.4f}, Val Loss: {val_loss_sum:.4f}"
            logging.info(log_message)
            print(log_message)
            torch.save({
                'step': step + 1,
                'model_state_dict':     tarModel.state_dict(),
                'aux_model_state_dict': auxModel.state_dict() if auxModel is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'train_loss':           train_loss_sum,
                'val_loss':             val_loss_sum,
                'val_task_accuracy':    val_task_accuracy
            }, os.path.join(save_path, f'model_checkpoint_step_{step+1}.pth'))
            
            del val_loss_component
            del val_loss_sum

        
        
        del task_loss_sum
        del train_loss_sum
        del aux_output
        del ua_output
        # if (step + 1) % 100 == 0:
        # torch.cuda.empty_cache()
    # writer.close()

if __name__ == "__main__":

    curr_placement = "WRIST"

    print("gpu state:", torch.cuda.is_available(), file=sys.stderr)
    if not torch.cuda.is_available():
        print("script done.")
        exit(0)

    train_datasets = {
        # "2012-OPPORTUNITY" :    ["acc", ],
        # "2012-realdisp":        ["acc", "gyro", "magnetic"],
        "2019-WISDM":           ["acc", "gyro"],
        # "2020-capture24":       ["acc",]
    }

    # batch_size = 1536 if args_global.MODEL_STRUCTURE == 'pfm50' else 2048
    batch_size = 128

    with open("optimal_params_0/optimal_params.json", "r") as f:
        optimal_params = json.load(f)

    TEST_SAMPLE_TIME_WINDOW = 6000
    PREFIX = f"{DATA_CATEGORY_PATH}/{curr_placement}"    

    # averaged in task number
    

    # prepare dataLoaders
    dataset_modality_taskName_tuple = []
    for d in train_datasets:
        for taskName in DATASET_TO_TASK[d]:
            taskLabelSet = grabTasklabels(curr_placement, d, TEST_SAMPLE_TIME_WINDOW, taskName)
            if len(taskLabelSet) == 1 and 0 in taskLabelSet:
                continue

            for m in train_datasets[d]:
                iterNum = 1
                if d == "2020-capture24" and taskName == 'annotation':
                    iterNum = 3
                dataset_modality_taskName_tuple += [(d, m, iterNum, taskName)]
    taskWeight = dict(Counter([(d, taskName) for d, _, _, taskName in dataset_modality_taskName_tuple]))

    perIter     = batch_size // sum([iterNum for _, _, iterNum, _ in dataset_modality_taskName_tuple])
    batch_size  = perIter * sum([iterNum for _, _, iterNum, _ in dataset_modality_taskName_tuple]) 
    # Size of the sample in use for each iteration.

    # Prepare the task and modality.
    dataLoaderList = []
    warmup_decay_train_params = []
    dataset_task_to_classifier = {}
    dataset_modality_to_encoder = {}

    # 
    maximum_taskNum = 0
    for d, m, iterNum, taskName in dataset_modality_taskName_tuple:
        datasetKey = f"{d}_{m}"
        TEST_SAMPLE_TIME_WINDOW = 6000
        if d == "2012-realdisp":
            TEST_SAMPLE_TIME_WINDOW = 2000

        curTimeLen, curTimeWindow = timeWindowSetting[optimal_params[datasetKey]["window_setting"]]

        if (d, taskName) not in dataset_task_to_classifier:
            taskLabelSize = len(grabTasklabels(curr_placement, d, TEST_SAMPLE_TIME_WINDOW, taskName)) - 1
            if maximum_taskNum < taskLabelSize:
                maximum_taskNum = taskLabelSize
            print("-- taskUpdata: ", taskName, taskLabelSize, file=sys.stderr)
            # dataset_task_to_classifier[(d, taskName)] = FwBlock(input_size=96 * TEST_SAMPLE_TIME_WINDOW // 200, output_size=len(grabTasklabels(curr_placement, d, TEST_SAMPLE_TIME_WINDOW, taskName)) - 1, hidden_params=[1024, ]).to(device)
            dataset_task_to_classifier[(d, taskName)] = ClassAvgPoolProjector(96, taskLabelSize).to(device)
            # for params in dataset_task_to_classifier[(d, taskName)].parameters():
            #     params.requires_grad = False # 
            
            if args_global.loss_style != 1:
                warmup_decay_train_params += [{"params": dataset_task_to_classifier[(d, taskName)].parameters()}]

        curMMAP_MODE = optimal_params[datasetKey]["MMAP_MODE"]

        # prepare dataset
        batch_size_d = perIter * iterNum # curr_batch_size of the dataLoader
        # self.EMBEDDING_MODEL = kwargs["EMBEDDING_MODEL"] if "EMBEDDING_MODEL" in kwargs else "all-mpnet-base-v2"
        EMBEDDING_MODEL = args_global.EMBEDDING_MODEL

        train_dataset = PretrainDataset(PREFIX, d,
                                        transform_x=ResampleTransform(timeLength=curTimeLen, timeWindow=curTimeWindow, method="nearest"),
                                        trainValSplit=None if not curMMAP_MODE else 'train',
                                        taskName=taskName,
                                        sampleTimeWindow=TEST_SAMPLE_TIME_WINDOW,
                                        sampleMiniWindow=True,
                                        verbose=0,
                                        targetFolders=TarFolderFilter(modality=[m,]),
                                        EXCLUDE_NONE = True,
                                        MMAP_MODE = curMMAP_MODE,
                                        SHUFFLE = curMMAP_MODE,
                                        HDF5=True,
                                        EMBEDDING_MODEL=EMBEDDING_MODEL,
                                        POST_SPLIT=0 if not curMMAP_MODE else -1
        ) # all users are used.
        
        print('dataset:', len(train_dataset), taskName, file=sys.stderr)
        
        train_loader = InfiniteDataLoader(DataLoader(train_dataset, batch_size=batch_size_d, pin_memory=True, num_workers=4, shuffle=not curMMAP_MODE), 
                                          shuffle_manually=curMMAP_MODE, 
                                          segmentation=args_global.DATASET_SEGMENTATION,
                                          use_prefetcher=True,
                                          device=device)
        print('dataloader:', len(train_loader.dataLoader), taskName, file=sys.stderr)
        
        val_dataset = PretrainDataset(PREFIX, d,
                                        transform_x=ResampleTransform(timeLength=curTimeLen, timeWindow=curTimeWindow, method="nearest"),
                                        trainValSplit=None if not curMMAP_MODE else 'val',
                                        taskName=taskName,
                                        sampleTimeWindow=TEST_SAMPLE_TIME_WINDOW,
                                        sampleMiniWindow=True,
                                        verbose=0,
                                        targetFolders=TarFolderFilter(modality=[m,]),
                                        EXCLUDE_NONE = True,
                                        MMAP_MODE = curMMAP_MODE,
                                        HDF5=True,
                                        EMBEDDING_MODEL=EMBEDDING_MODEL,
                                        POST_SPLIT=1 if not curMMAP_MODE else -1
        )
        val_loader = DataLoader(val_dataset, batch_size=128, pin_memory=True, num_workers=8, shuffle=False)

        # load encoders
        #   encoders can be retrained.
        # encoders from AE model for now.
        with open(f"json_output/{d}.json", "r") as f:
            encoder_info = json.load(f)[m]
            conv_params = encoder_info["conv_params"]        

        # hidden size = 96
        data_feature_extractor = LinearFeatureExtractor2(conv_params[0][0], curTimeLen, 96) # use linear projector instead.

        optimizer = Adam(data_feature_extractor.parameters(), lr=1e-3, weight_decay=1e-6, amsgrad=True) # tuning parameters

        data_feature_extractor.to(device)
        data_feature_extractor.eval()

        if (d, m) not in dataset_modality_to_encoder:
            dataset_modality_to_encoder[(d, m)] = [data_feature_extractor, optimizer]

        # prepare data
        dataLoaderList += [(d, m, taskName, iter(train_loader), val_loader, torch.tensor(1.0 / taskWeight[(d, taskName)], dtype=torch.float32).to(device), len(train_loader.dataLoader))]

    seq_len                 = TEST_SAMPLE_TIME_WINDOW // 200

    models = []

    ############################### load w2vBERT model ###############################
    model_type =        args_global.MODEL_STRUCTURE
    path2wpf =          args_global.MODEL_FOLDER
    trainMechanism =    args_global.MODEL_TRAINING_STRATEGY
    trainStep =         args_global.MODEL_VERSION_IN_STEP

    with open(f"{path2wpf}/{trainMechanism}/_metaInfo.json", "r") as f:
        cfg = AttrDict(json.load(f))

    print(cfg.keys())

    cfg.quantizer_cfg = AttrDict(cfg.quantizer_cfg)
    tarModel = WearablePretrainedEncoder(   cfg.TEST_SAMPLE_TIME_WINDOW // 200, 
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
    if torch.cuda.is_available():
        dataDict= AttrDict(torch.load(f"{path2wpf}/{trainMechanism}/history/model_checkpoint_step_{trainStep}.pth"))
    else:
        dataDict= AttrDict(torch.load(f"{path2wpf}/{trainMechanism}/history/model_checkpoint_step_{trainStep}.pth", map_location=torch.device('cpu')))

    if args_global.loss_style != 4:
        tarModel.load_state_dict(dataDict.model_state_dict) 
    # freeze quantizer
    tarModel.freeze_quantizer()
    models  += [tarModel.to(device)]


    # w2vBERT auxModel
    auxModel = FwBlock(cfg.hidden_dim, int(cfg.quantizer_cfg.codebook_num_groups * cfg.quantizer_cfg.codebook_num_vars), 
                        hidden_params=[144, ])
    
    if args_global.loss_style != 4:
        auxModel.load_state_dict(dataDict.aux_model_state_dict)

    models  += [auxModel.to(device)]
    ############################### load w2vBERT model end  ###############################

    ############################### create w2vBERT model    ############################### 
    universal_alignment_aux_model = FwBlock(cfg.hidden_dim, EMBEDDING_to_SHAPE[EMBEDDING_MODEL], hidden_params=[1024, ])
    models += [universal_alignment_aux_model.to(device)]

    ############################### ############################### ###############################
    warmup_decay_train_params += [{"params": filter(lambda x: x.requires_grad, tarModel.parameters())},
                                  {"params": auxModel.parameters()}, # depends
                                  {"params": universal_alignment_aux_model.parameters()}
                                ]

    # Parameters for the learning rate schedule
    init_lr         = 0.0001 
    peak_lr         = 0.0005
    final_lr        = 0.0001
    warm_up_step    = 5000 # total iterations: 300000 * 0.1
    decay_factor    = 0.99993 
    
    # Create optimizer and scheduler
    optimizer, scheduler = set_warmup_and_decay_lr(warmup_decay_train_params, init_lr, peak_lr, final_lr, warm_up_step, decay_factor, amsgrad=True)
    # optimizer = Adam(warmup_decay_train_params, lr=1e-3, weight_decay=1e-6) # tuning parameters
    # scheduler = None
    # Training settings
    total_steps     = 50000 # 300000 batches

    # save the meta information
    # exp_name = f"exp-full-dropout-real-randFeat-{'MSE' if args_global.UA_LOSS_TYPE == 0 else 'SoftDTW'}-{EMBEDDING_MODEL_SHORTCUT[EMBEDDING_MODEL]}"
    exp_name = f"exp-full-dropout-{batch_size}-real{('-'+ model_type) if model_type != 'pfm10' else ''}-{'MSE' if args_global.UA_LOSS_TYPE == 0 else 'SoftDTW'}-{EMBEDDING_MODEL_SHORTCUT[EMBEDDING_MODEL]}-ext"
    if "-loaderLength" in trainMechanism:
        exp_name += "-loaderLength"
    
    loss_style_name = {
        0: "all",
        1: "noTask",
        2: "noUA",
        3: "noMLM",
        4: "noPretrained"
    }
    metaInfo = {
        "exp_name": exp_name,
        
        "pretrained_cfg": {
            "model_type" :      model_type,
            "path2wpf":         path2wpf,
            "trainMechanism":   trainMechanism,
            "trainStep":        trainStep,
        },

        "segmentation":     args_global.DATASET_SEGMENTATION,

        "train_datasets":   train_datasets,
        "batch_size":       batch_size,
        "total_steps":      total_steps,

        "quantizer_cfg": {
            "disabled":     None,
        },
        'encoder_learning_rate': 1e-3,

        "warmup_decay_cfg": {
            "init_lr":      init_lr,
            "peak_lr":      peak_lr,
            "final_lr":     final_lr,
            "warm_up_step": warm_up_step,
            "decay_factor": decay_factor,
            "AMSGRAD" :     True
        },

        "universal_alignment_loss_style": "MSE" if args_global.UA_LOSS_TYPE == 0 else "SoftDTW",
        "loss_style": loss_style_name[args_global.loss_style]

    }

    parentFolder = f"results{('_%s' % loss_style_name[args_global.loss_style]) if args_global.loss_style != 0 else ''}/{exp_name}"
    # parentFolder = f"results//{exp_name}"
    
    print(parentFolder)
    print(exp_name)
    os.makedirs(parentFolder, exist_ok=True)
    with open(os.path.join(parentFolder, "_metaInfo.json"), "w") as f:
        json.dump(metaInfo, f, indent=4)

    writer = SummaryWriter(log_dir=f"{parentFolder}/log")
    try:
        post_train_model(models, dataLoaderList, 
                        dataset_modality_to_encoder,
                        dataset_task_to_classifier,
                        optimizer, scheduler, batch_size, total_steps, 
                universal_alignment_loss_style=args_global.UA_LOSS_TYPE,
                device=device, 
                save_path=f"{parentFolder}/history", 
                log_path=f"{parentFolder}/log",
                writer=writer,
                maximum_taskNum = maximum_taskNum
                )
        
    except Exception as e:
        print(e, file=sys.stderr)
        traceback.print_exc()
    finally:
        writer.close()
    
    print("script finished.")


