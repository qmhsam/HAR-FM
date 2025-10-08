# 
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import logging
import json
from utils import *
from models import grabClassifier, AutoEncoder, WearablePretrainedEncoder, FwBlock

import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter


import argparse
# Changed from:     train_1_pretrainedModel-contrastiveLossTopk_bs1024_codebook_amsgrad.py
#   README:         change the contrastive learning loss computation to top k elements with a stride of 3 << 600 ms

parser = argparse.ArgumentParser(description='pretraining foundation model')
parser.add_argument('--TRAIN_SOURCE', default=False, action="store_true", help='if train the low-level dataset-wise encoder')

parser.add_argument('--PFM_VERSION', default='pfm5', type=str, help='version of the PFM, mainly different in size', choices=['pfm5', 'pfm10', 'pfm18', 'pfm34', 'pfm50'])
parser.add_argument('--SHARE_TRANSFORMER_LAYERS', default=False, action="store_true")     
parser.add_argument('--AMSGRAD', default=False, action="store_true")     

parser.add_argument('--CODEBOOK_NUM_GROUPS', default=4, type=int, choices=[1,   2, 3,  4,  6, 8, 10])
parser.add_argument('--CODEBOOK_NUM_VARS', default=8, type=int, choices=[4096, 64, 16, 8,  4])

parser.add_argument('--LEARNING_RATE_DEFACTOR', default=1, type=int, choices=[1, 2, 5, 10])   
parser.add_argument('--REWEIGHT_MLM',   default=1.0,    type=float, choices=[1., 0.5, 0.1])   
parser.add_argument('--REWEIGHT_CL',    default=1.0,    type=float, choices=[1., 0.8, 0.1])   


parser.add_argument('--DATASET_SEGMENTATION', default=15, type=int, choices=[15, 25, 150])   


parser.add_argument('--LOSS_TYPE', default=0, type=int, choices=[0, 1, 2, 3])   
LOSS_TYPE_NAME = ["v2wBERT", "reg-v2wBERT", "BERT", "wav2vec2"]
# DESC.
# @LOSS_TYPE:   contrastive  |  masked language model (CE) | masked language model (MSE)    |   Similar Work            |
#   0       :   o            |  o                          | x                              |   v2wBERT                 |
#   1       :   o            |  x                          | o                              |   v2wBERT+regression MLM  |
#   2       :   x            |  x                          | o                              |   BERT                    |
#   3       :   o            |  x                          | x                              |   wav2vec2.0              |

args_global = parser.parse_args()



def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


timeWindowSetting = {
    "30hz":     (6, 200),
    "50hz":     (10, 200),
    "100hz":    (20, 200)
} 

device = get_device()
def compute_loss(feat_, aux_feat_, masked_pos, info_, loss_style, mlm_weight=1.0, return_mlm_accuracy=False):

    # Calculate contrastive and perplexity loss components
    if loss_style != 3:
        pred_ = info_["context_encoded"] 
    else:
        pred_ = feat_ # take full backbone as the contextual encoder.
    l_c = torch.tensor(0.0, device=masked_pos.device).float()
    l_p = torch.tensor(0.0, device=masked_pos.device).float()
    if loss_style != 2:
        l_c = cosine_similarity_loss_topk(pred_, info_["quantized_vector"], masked_pos, stride=2, topk=10) # 5 -> 1 sec away
        l_p = (info_["num_vars"] - info_["prob_perplexity"]) / info_["num_vars"]

    L_1 = l_c + l_p * 0.1

    # Calculate MLM loss based on the specified style
    loss_mlm = torch.tensor(0.0, device=masked_pos.device).float()
    if loss_style == 0:
        mlm_criterion = nn.CrossEntropyLoss()
        pred_   = aux_feat_
        output_ = pred_[masked_pos.bool()]
        target_ = info_["quantized_ids"][masked_pos.bool()]
        partial_dim = output_.shape[1] // tarModel.codebook_num_groups
        loss_mlm    = torch.stack([mlm_criterion(output_[:, i * partial_dim: (i + 1) * partial_dim], target_[:, i])
                               for i in range(tarModel.codebook_num_groups)]).mean()
        
        # target_ids  = info_["quantized_ids"]
        # b, t ,g = target_ids.shape
        # target_ids  = target_ids.reshape(-1, target_ids.shape[-1])
        # split_len   = aux_output.shape[-1] // target_ids.shape[-1]

        # mlm_loss    = torch.stack([mlm_criterion(aux_output[:, i * split_len: (i + 1) * split_len], 
        #                                          target_ids[:, i]) 
        #                             for i in range(target_ids.shape[-1])]).mean()
        if return_mlm_accuracy:
            prediction = output_.view(-1, target_.shape[-1], partial_dim).argmax(-1)
            correct_mlm = (prediction == target_).all(-1).sum().float().item() / prediction.shape[0]

    elif loss_style in [1, 2]:
        mlm_criterion = nn.MSELoss()
        pred_ = aux_feat_
        output_ = pred_[masked_pos.bool()]
        target_ = info_["input_embedding"][masked_pos.bool()]
        loss_mlm = mlm_criterion(output_, target_)

    L_2 = loss_mlm
    if loss_style == 2:
        total_loss = L_2

    elif loss_style == 3:
        total_loss = L_1

    else:
        total_loss = L_1 * args_global.REWEIGHT_CL + L_2 * mlm_weight

    if return_mlm_accuracy:
        return total_loss, l_c, l_p, loss_mlm, correct_mlm
    
    return total_loss, l_c, l_p, loss_mlm

def pretrain_model(models, dataLoaderList, 
                   optimizer, scheduler, warm_up_step,
                   batch_size,
                   num_steps, 
                   mask_cfg, 
                   valid_mask,
                   loss_style,  # [0, 1, 2, 3]: 1, 2 -> MLM (MSE loss), 0 -> w2vBERT(CE loss)
                   device, 
                   save_path="./results/undefined", 
                   log_path="./results/undefined/log", mlm_weight=args_global.REWEIGHT_MLM):
    assert len(models) in [1, 2]
    tarModel = models[0]
    auxModel = models[1] if len(models) > 1 else None
    
    # Set up logging and tensorboard
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_path, 'training.log'), level=logging.INFO)
    writer = SummaryWriter(log_dir=log_path)
    weight_denom = torch.tensor(sum([k[-1] for k in dataLoaderList]), device=device)

    print(weight_denom.item(), file=sys.stderr)
    for step in range(num_steps):
        tarModel.train()
        if auxModel is not None:
            auxModel.train()

        train_loss_sum = 0.0

        loss_component = {
            "contrastive_loss": 0.0,
            "perplexity_loss" : 0.0,
            "mlm_loss": 0.0
        }

        extra_optimizers = []

        for d, m, _, train_dLoader, val_dLoader, encoders, optimizers, loader_length in tqdm(dataLoaderList):
            # print(d, m, file=sys.stderr)
            batch = next(train_dLoader)
            data, _, metaInfo = batch
            bsz, tsz, csz, fsz = data.shape

            data = data.reshape(-1, csz, fsz)

            for i in range(len(encoders)):

                if args_global.TRAIN_SOURCE and step + 1 > 0:
                    encoders[i].train()
                else:
                    encoders[i].eval()
                

            encoder = encoders[0]
            if len(encoders) != 1:
                # TODO: support certain optimizer attached.
                encoder = random.sample(encoders, 1)[0]
                if args_global.TRAIN_SOURCE and step + 1 > 0:
                    extra_optimizers += [optimizers[0],]
                    # TODO: implement the training logic in some future
            elif args_global.TRAIN_SOURCE and step + 1 > 0:
                extra_optimizers += [optimizers[0],]
            
                
            dataset_wise_feat = encoder(data)
            dataset_wise_feat = dataset_wise_feat.reshape(bsz, tsz, -1)
            # print(dataset_wise_feat.shape, file=sys.stderr)
            masked_pos = generate_span_mask(bsz, tsz, device=device, **mask_cfg)

            DATASET_FEAT_DROPOUT_WEIGHT_1 = F.dropout(torch.ones(dataset_wise_feat.shape[-1], device=device), 0.2)
            DATASET_FEAT_DROPOUT_WEIGHT_2 = F.dropout(torch.ones(dataset_wise_feat.shape[-1], device=device), 0.5)
            
            # loss 1
            feat_, info_ = tarModel(F.layer_norm(dataset_wise_feat * DATASET_FEAT_DROPOUT_WEIGHT_1, (dataset_wise_feat.shape[-1], )), masked_pos)
            feat_aug_, info_aug_ = tarModel(F.layer_norm(dataset_wise_feat * DATASET_FEAT_DROPOUT_WEIGHT_2, (dataset_wise_feat.shape[-1], )), masked_pos)

            # print(feat_.shape, file=sys.stderr)
            # print(info_.keys(), file=sys.stderr)
            # print(info_["context_encoded"].shape, file=sys.stderr)
            # print(info_["quantized_vector"].shape, file=sys.stderr)
            # print(info_["quantized_ids"].shape, file=sys.stderr)

            feat_cat = torch.concat([feat_, feat_aug_])
            masked_pos_cat = masked_pos.repeat(2, 1)
            info_cat = {
                "context_encoded":  torch.concat([info_["context_encoded"],     info_aug_["context_encoded"]]) if loss_style != 3 else None,
                "quantized_vector": torch.concat([info_["quantized_vector"],    info_aug_["quantized_vector"]]),
                "num_vars":         info_["num_vars"],
                "prob_perplexity":  info_["prob_perplexity"] - info_aug_["num_vars"] + info_aug_["prob_perplexity"],
                "quantized_ids":    torch.concat([info_["quantized_ids"],       info_aug_["quantized_ids"]])    if loss_style == 0 else None,
                "input_embedding":  torch.concat([info_["input_embedding"],     info_aug_["input_embedding"]])  if loss_style in [1, 2] else None
            }

            aux_feat_cat = auxModel(feat_cat) if auxModel is not None else None
            total_loss, l_c, l_p, loss_mlm = compute_loss(feat_cat, aux_feat_cat, masked_pos_cat, info_cat, loss_style, mlm_weight=mlm_weight)

            dataset_loss_weight = loader_length / weight_denom

            train_loss_sum += total_loss * bsz * 2 * dataset_loss_weight
            
            loss_component["contrastive_loss"]  += l_c.item() * bsz * 2 * dataset_loss_weight
            loss_component["perplexity_loss"]   += l_p.item() * bsz * 2 * dataset_loss_weight
            loss_component["mlm_loss"]          += loss_mlm.item() * bsz * 2 * dataset_loss_weight

            print(d, m, ":", dataset_loss_weight.item(), file=sys.stderr)

            # aux_feat_ = auxModel(feat_) if auxModel is not None else None
            # aux_feat_aug_ = auxModel(feat_aug_) if auxModel is not None else None
            # context_encoded
            # quantized_vector
            #   (info_["num_vars"] - info_["prob_perplexity"]) / info_["num_vars"]
            #       num_vars
            #       prob_perplexity
            # quantized_ids
            # input_embedding

            # total_loss, l_c, l_p, loss_mlm = compute_loss(feat_, aux_feat_, masked_pos, info_, loss_style, mlm_weight=mlm_weight)

            # train_loss_sum += total_loss * bsz
            
            # loss_component["contrastive_loss"]  += l_c.item() * bsz
            # loss_component["perplexity_loss"]   += l_p.item() * bsz
            # loss_component["mlm_loss"]          += loss_mlm.item() * bsz

            # # aug_loss
            # total_loss, l_c, l_p, loss_mlm = compute_loss(feat_aug_, aux_feat_aug_, masked_pos, info_aug_, loss_style, mlm_weight=mlm_weight)
                        
            # train_loss_sum += total_loss * bsz
            
            # loss_component["contrastive_loss"]  += l_c.item() * bsz
            # loss_component["perplexity_loss"]   += l_p.item() * bsz
            # loss_component["mlm_loss"]          += loss_mlm.item() * bsz

            # CONTEXT_FEAT_DROPOUT_WEIGHT = F.dropout(torch.ones(feat_.shape[-1], device=device), 0.3)
            # aux_feat_ = auxModel(F.layer_norm(feat_ * CONTEXT_FEAT_DROPOUT_WEIGHT, (feat_.shape[-1],))) if auxModel is not None else None
            
            # total_loss, l_c, l_p, loss_mlm = compute_loss(feat_, aux_feat_, masked_pos, info_, loss_style, mlm_weight=mlm_weight)
            
            # train_loss_sum += total_loss * bsz
            
            # loss_component["contrastive_loss"]  += l_c.item() * bsz
            # loss_component["perplexity_loss"]   += l_p.item() * bsz
            # loss_component["mlm_loss"]          += loss_mlm.item() * bsz


        train_loss_sum /= batch_size * 2 # reweight 

        loss_component["contrastive_loss"]  /= batch_size * 2
        loss_component["perplexity_loss"]   /= batch_size * 2
        loss_component["mlm_loss"]          /= batch_size * 2

        # log each step
        writer.add_scalar('Train/loss_sum',  train_loss_sum.item(), step)
        for name, value in loss_component.items():
            writer.add_scalar(f'Train/{name}', value, step)

        # zero_grad
        optimizer.zero_grad()
        if extra_optimizers:
            for opt in extra_optimizers:
                opt.zero_grad()

        # backward
        train_loss_sum.backward()

        # step
        optimizer.step()
        for extra_optimizer in extra_optimizers:
            extra_optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        if (step + 1) % 5000 == 0 or step == 0:
            tarModel.eval()
            if auxModel is not None:
                auxModel.eval()
            # TODO: validation computation            
            val_loss_sum = 0.0

            val_loss_component = {
                "contrastive_loss": 0.0,
                "perplexity_loss" : 0.0,
                "mlm_loss": 0.0
            }
            avg_mlm_acc = 0.0
            for d, m, _, _, val_dLoader, encoders, _, _ in dataLoaderList:
                dm_total_size= len(val_dLoader.dataset)
                print(d, m, f": {dm_total_size}", file=sys.stderr)
                dm_val_loss_sum, dm_l_c, dm_l_p, dm_l_mlm = 0., 0., 0., 0.
                
                dLoader = tqdm(DataPrefetcherThread(val_dLoader, device=device), desc=f'Step {step+1}/{num_steps} [Valid on {d:>12s}_{m:9s}]', leave=False, mininterval=5)

                mlm_accuracy_sum = 0.0
                for batch in dLoader:
                    data, _, metaInfo = batch
                    bsz, tsz, csz, fsz = data.shape

                    data = data.reshape(-1, csz, fsz)
                    
                    # if args_global.TRAIN_SOURCE:
                    for i in range(len(encoders)):
                        encoders[i].eval()
                            
                    encoder = encoders[0] if len(encoders) == 1 else random.sample(encoders, 1)[0]
                    with torch.no_grad():
                        dataset_wise_feat = encoder(data)
                        dataset_wise_feat = dataset_wise_feat.reshape(bsz, tsz, -1)
                        
                        masked_pos = valid_mask.expand(bsz, tsz) if valid_mask is not None else generate_span_mask(bsz, tsz, device=device, **mask_cfg)
                        feat_, info_ = tarModel(dataset_wise_feat, masked_pos)

                        aux_feat_ = auxModel(feat_) if auxModel is not None else None
                        mlm_accuracy = -1e-6
                        if loss_style == 0:
                            total_loss, l_c, l_p, loss_mlm, mlm_accuracy = compute_loss(feat_, aux_feat_, masked_pos, info_, loss_style, mlm_weight=mlm_weight, return_mlm_accuracy=True)
                        else:
                            total_loss, l_c, l_p, loss_mlm = compute_loss(feat_, aux_feat_, masked_pos, info_, loss_style, mlm_weight=mlm_weight)

                        
                    mlm_accuracy_sum += mlm_accuracy * bsz 

                    dm_val_loss_sum += total_loss.item() * bsz
                    # 
                    dm_l_c      += l_c.item() * bsz
                    dm_l_p      += l_p.item() * bsz
                    dm_l_mlm    += loss_mlm.item() * bsz
                
                dm_val_loss_sum /= dm_total_size
                # 
                dm_l_c          /= dm_total_size
                dm_l_p          /= dm_total_size
                dm_l_mlm        /= dm_total_size

                mlm_accuracy_sum /= dm_total_size
                avg_mlm_acc += mlm_accuracy_sum / len(dataLoaderList)

                # log the validation for each dataset.
                loss_component_local = {
                    "contrastive_loss": dm_l_c,
                    "perplexity_loss" : dm_l_p,
                    "mlm_loss":         dm_l_mlm
                }
                writer.add_scalar(f'Valid/loss_sum/{d}_{m}',  dm_val_loss_sum, step)
                for name, value in loss_component_local.items():
                    writer.add_scalar(f'Valid/{name}/{d}_{m}', value, step)

                # update the summary
                val_loss_sum                            += dm_val_loss_sum

                val_loss_component["contrastive_loss"]  += dm_l_c
                val_loss_component["perplexity_loss"]   += dm_l_p
                val_loss_component["mlm_loss"]          += dm_l_mlm


            denom   =   len(dataLoaderList)
            val_loss_sum                                /= denom # reweight 

            val_loss_component["contrastive_loss"]      /= denom
            val_loss_component["perplexity_loss"]       /= denom
            val_loss_component["mlm_loss"]              /= denom
            
            writer.add_scalar(f'Valid/loss_sum/Avg',   val_loss_sum, step)
            for name, value in val_loss_component.items():
                writer.add_scalar(f'Valid/{name}/Avg',  value, step)

            log_message = f"Step {step + 1}, Train Loss: {train_loss_sum:.4f}, Val Loss: {val_loss_sum:.4f}, {'random' if valid_mask is None else 'fixed'} Mask Acc: {avg_mlm_acc:.4f}"
            logging.info(log_message)
            print(log_message)
            torch.save({
                'step': step + 1,
                'model_state_dict':     tarModel.state_dict(),
                'aux_model_state_dict': auxModel.state_dict() if auxModel is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'train_loss':           train_loss_sum,
                'val_loss':             val_loss_sum
            }, os.path.join(save_path, f'model_checkpoint_step_{step+1}.pth'))
            
        if (step + 1) % 10 == 0:
            torch.cuda.empty_cache()
            
    writer.close()

if __name__ == "__main__":

    curr_placement = "WRIST"


    train_datasets = {
        # "2012-OPPORTUNITY" :    ["acc", ],
        # "2012-realdisp":        ["acc", "gyro", "magnetic"],
        "2019-WISDM":           ["acc", "gyro"],
        # "2020-capture24":       ["acc",]
    }

    batch_size = 128
    # batch_size = 128 * 8

    with open("optimal_params_0/optimal_params.json", "r") as f:
        optimal_params = json.load(f)
        

    # prepare dataLoaders
    dataset_modality_tuple = []
    for d in train_datasets:
        iterNum = 1 
        if d in ["2020-capture24", ]: # number of iteration in used, special iteration number for larger datasets.
            iterNum = 2
        for m in train_datasets[d]:
            dataset_modality_tuple += [(d, m, iterNum)]

    perIter     = batch_size // sum([iterNum for _, _, iterNum in dataset_modality_tuple])

    batch_size  = perIter * sum([iterNum for _, _, iterNum in dataset_modality_tuple]) # Size of the sample in use for each iteration.

    dataLoaderList = []

    PREFIX = f"{DATA_CATEGORY_PATH}/{curr_placement}"
    TEST_SAMPLE_TIME_WINDOW = 6000

    warmup_decay_train_params = []
    models = []
    for d, m, iterNum in dataset_modality_tuple:
        taskName = DATASET_TO_TASK[d][DEFAULT_GRAINED_TASK]
        datasetKey = f"{d}_{m}"
        curTimeLen, curTimeWindow = timeWindowSetting[optimal_params[datasetKey]["window_setting"]]
        curMMAP_MODE = optimal_params[datasetKey]["MMAP_MODE"]

        # prepare dataset
        batch_size_d = perIter * iterNum # curr_batch_size of the dataLoader
        train_dataset = PretrainDataset(PREFIX, d,
                                        transform_x=ResampleTransform(timeLength=curTimeLen, timeWindow=curTimeWindow, method="nearest"),
                                        trainValSplit="train",
                                        taskName=taskName,
                                        sampleTimeWindow=TEST_SAMPLE_TIME_WINDOW,
                                        sampleMiniWindow=True,
                                        verbose=0,
                                        targetFolders=TarFolderFilter(modality=[m,]),
                                        EXCLUDE_NONE = False,
                                        MMAP_MODE = curMMAP_MODE,
                                        SHUFFLE = curMMAP_MODE,
                                        HDF5=True
        ) # all users are used.
        train_loader = InfiniteDataLoader(DataLoader(train_dataset, batch_size=batch_size_d, pin_memory=False, num_workers=4, shuffle=not curMMAP_MODE), 
                                          shuffle_manually=curMMAP_MODE, 
                                          segmentation=args_global.DATASET_SEGMENTATION,
                                          use_prefetcher=True,
                                          device=device)
        
        val_dataset = PretrainDataset(PREFIX, d,
                                        transform_x=ResampleTransform(timeLength=curTimeLen, timeWindow=curTimeWindow, method="nearest"),
                                        trainValSplit="val",
                                        taskName=taskName,
                                        sampleTimeWindow=TEST_SAMPLE_TIME_WINDOW,
                                        sampleMiniWindow=True,
                                        verbose=0,
                                        targetFolders=TarFolderFilter(modality=[m,]),
                                        EXCLUDE_NONE = False,
                                        MMAP_MODE = curMMAP_MODE,
                                        HDF5=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size_d, pin_memory=False, num_workers=4)

        # load encoders
        #   encoders can be retrained.
        # encoders from AE model for now.
        with open(f"json_output/{d}.json", "r") as f:
            encoder_info = json.load(f)[m]
            conv_params = encoder_info["conv_params"]

        # fetch model structure
        model = grabClassifier(conv_params, input_size=curTimeLen, output_size=len(grabTasklabels(curr_placement, d, TEST_SAMPLE_TIME_WINDOW, taskName)) - 1)
        feature_extractor = model.feature

        # # Replace the feature extractor in the classifier
        optimizer = Adam(feature_extractor.parameters(), lr=1e-5, weight_decay=1e-6) # tuning parameters

        feature_extractor.to(device)
        feature_extractor.eval()

        # prepare data
        dataLoaderList += [(d, m, taskName + ".Pretrained", iter(train_loader), val_loader, [feature_extractor, ], [optimizer,], len(train_loader.dataLoader))]

    # prepare the model
    # codebook_num_groups     = 4     # 1, 4
    # codebook_num_vars       = 8     # 32, 96

    codebook_num_groups     = args_global.CODEBOOK_NUM_GROUPS     # 1, 4
    codebook_num_vars       = args_global.CODEBOOK_NUM_VARS       # 32, 96

    # codebook_vq_dim         = 64    # 
    codebook_vq_dim         = 16 * codebook_num_groups # var_dim = 16

    quantizer_training_temp = (2, 0.5, 0.999993) #  0.99998
    use_trainable_mask      = True
    #   feature_size: int,
    #   hidden_dim: int,

    feature_size            = 96
    hidden_dim              = 96
    seq_len                 = TEST_SAMPLE_TIME_WINDOW // 200
    tarModel = WearablePretrainedEncoder(   seq_len, 
                                            feature_size, 
                                            hidden_dim,
                                            use_quantizer           = True,
                                            codebook_num_groups     = codebook_num_groups,
                                            codebook_num_vars       = codebook_num_vars,
                                            codebook_vq_dim         = codebook_vq_dim,
                                            quantizer_training_temp = quantizer_training_temp,
                                            use_trainable_mask      = use_trainable_mask,
                                            share_layers_t1         = args_global.SHARE_TRANSFORMER_LAYERS,
                                            share_layers_t2         = args_global.SHARE_TRANSFORMER_LAYERS,
                                            version                 = args_global.PFM_VERSION
                                        )
    warmup_decay_train_params += [{"params": tarModel.parameters()}]
    models  += [tarModel.to(device)]

    auxBlock = {
        "MLM":      FwBlock(hidden_dim, hidden_dim, 
                            hidden_params=[144, ]),
        "w2vBERT":  FwBlock(hidden_dim, int(codebook_num_groups * codebook_num_vars), 
                            hidden_params=[144, ])
    }

    print(f"LOSS_TYPE: {LOSS_TYPE_NAME[args_global.LOSS_TYPE]}")
    if args_global.LOSS_TYPE in [1, 2]:
        auxModel = auxBlock["MLM"].to(device)
        models  += [auxModel]
        warmup_decay_train_params += [{"params": auxModel.parameters()}]
    elif args_global.LOSS_TYPE in [0,]:
        auxModel = auxBlock["w2vBERT"].to(device)
        models  += [auxModel]
        warmup_decay_train_params += [{"params": auxModel.parameters()}]
    
    # Parameters for the learning rate schedule
    init_lr         = 0.0001 if args_global.LOSS_TYPE != 0 else 0.00001
    peak_lr         = (0.001 if args_global.LOSS_TYPE != 0 else 0.0001) / args_global.LEARNING_RATE_DEFACTOR
    final_lr        = 0.00001
    warm_up_step    = 20000 # total iterations: 300000 * 0.1
    decay_factor    = 0.99998 # 
    
    # Create optimizer and scheduler
    optimizer, scheduler = set_warmup_and_decay_lr(warmup_decay_train_params, init_lr, peak_lr, final_lr, warm_up_step, decay_factor, amsgrad=args_global.AMSGRAD)
    # optimizer = Adam(warmup_decay_train_params, lr=1e-3, weight_decay=1e-6) # tuning parameters
    # scheduler = None
    # Training settings
    total_steps     = 200000 # 300000 batches

    # # MASK_SETTING
    # batch_size: Any,
    # seq_len: Any,
    # max_gram: int = 3,
    # p: float = 0.2,
    # goal_num_predict: int = 15,

    mask_cfg = {
        "max_gram":         3,
        "p":                0.2,
        "goal_num_predict": 10
    }

    # cached valid mask
    fixed_valid_mask = False #coarse-1
    os.makedirs("_temp/", exist_ok=True)
    valid_mask_pth = "_temp/valid_mask.pt"
    if os.path.exists(valid_mask_pth):
        valid_mask = torch.load(valid_mask_pth)
    else:
        valid_mask = generate_span_mask(1, seq_len, device=device, **mask_cfg)
        torch.save(valid_mask, valid_mask_pth)
    if not fixed_valid_mask:
        valid_mask = None

    # save the meta information
    exp_name = f"exp-coarse-{0 if not args_global.SHARE_TRANSFORMER_LAYERS else 1}-{args_global.PFM_VERSION}-{LOSS_TYPE_NAME[args_global.LOSS_TYPE]}-loaderLength"
    metaInfo = {
        "exp_name": exp_name,
        "TEST_SAMPLE_TIME_WINDOW": TEST_SAMPLE_TIME_WINDOW,
        "segmentation": args_global.DATASET_SEGMENTATION,

        "train_datasets": train_datasets,
        "batch_size": batch_size,
        "total_steps": total_steps,

        "quantizer_cfg": {
            "codebook_num_groups": codebook_num_groups,
            "codebook_num_vars": codebook_num_vars,
            "codebook_vq_dim": codebook_vq_dim,
            "quantizer_training_temp": list(quantizer_training_temp),
        },

        "warmup_decay_cfg": {
            "init_lr":      init_lr,
            "peak_lr":      peak_lr,
            "final_lr":     final_lr,
            "warm_up_step": warm_up_step,
            "decay_factor": decay_factor,
            "AMSGRAD" :args_global.AMSGRAD
        },

        "feature_size":         feature_size,
        "hidden_dim":           hidden_dim,
        "use_trainable_mask":   use_trainable_mask,
        
        "shared_layers":    args_global.SHARE_TRANSFORMER_LAYERS,

        "cosine_similarity_stride": 2, #
        "cosine_similarity_type":  "topk",

        "mask_cfg":     mask_cfg,
        "mlm_weight":   args_global.REWEIGHT_MLM,
        "cl_weight":    args_global.REWEIGHT_CL,
        "valid_mask":   valid_mask.squeeze(0).tolist() if fixed_valid_mask else None,

        "LOSS_TYPE":    LOSS_TYPE_NAME[args_global.LOSS_TYPE],
    }

    parentFolder = f"results{f'-seg{args_global.DATASET_SEGMENTATION}' if args_global.DATASET_SEGMENTATION != 15 else ''}/g{codebook_num_groups}v{codebook_num_vars}{'-amsgrad' if args_global.AMSGRAD else ''}-topK{'-UD-step0' if args_global.TRAIN_SOURCE else ''}-lr{args_global.LEARNING_RATE_DEFACTOR}{'' if args_global.REWEIGHT_MLM == 1.0 else '-mlm' + str(int(1 / args_global.REWEIGHT_MLM))}{'' if args_global.REWEIGHT_CL == 1.0 else '-cl' + str(int(10 * args_global.REWEIGHT_CL))}/{exp_name}"

    # parentFolder = f"results//{exp_name}"
    os.makedirs(parentFolder, exist_ok=True)
    with open(os.path.join(parentFolder, "_metaInfo.json"), "w") as f:
        json.dump(metaInfo, f, indent=4)
    try:
        pretrain_model(models, dataLoaderList, 
                    optimizer, scheduler, warm_up_step,
                    batch_size,
                    total_steps, 
                    mask_cfg, 
                    valid_mask,
                    args_global.LOSS_TYPE,  # [0, 1, 2, 3]: 1, 2 -> MLM (MSE loss), 0 -> w2vBERT(CE loss)
                    device, 
                    save_path=f"{parentFolder}/history", 
                    log_path=f"{parentFolder}/log", 
                    mlm_weight=args_global.REWEIGHT_MLM)
        
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)  
    print("script finished.")


