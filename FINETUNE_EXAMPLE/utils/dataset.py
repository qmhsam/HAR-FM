from .serialization import decode_from_string
from .config import DATA_CATEGORY_PATH, DATASET_TO_TASK, DATA_CATEGORY_PATH, DEFAULT_GRAINED_TASK

import itertools
import time

import dill as pickle
from functools import reduce
from pprint import pprint
import os, copy
from sys import getsizeof as getsize

import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset

from collections import OrderedDict
# from concurrent.futures import ThreadPoolExecutor

import sys
import h5py

import random
import numpy as np
import torch

from collections import deque
from sklearn.model_selection import train_test_split
import threading

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# import asyncio

def set_seed(seed):
    """
    Set the seed for random number generation for reproducibility.
    
    Parameters:
    - seed: The seed value to set for random number generation.
    """
    random.seed(seed)                      # Set seed for random library
    np.random.seed(seed)                   # Set seed for NumPy
    torch.manual_seed(seed)                 # Set seed for PyTorch
    torch.cuda.manual_seed(seed)            # Set seed for CUDA (if using GPU)
    torch.cuda.manual_seed_all(seed)        # Set seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False     # Disable CuDNN benchmarking
    
def get_size(obj, seen=None):
    """Recursively calculate size of objects including NumPy arrays."""
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum((get_size(key, seen) + get_size(value, seen) for key, value in obj.items()))
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_size(item, seen) for item in obj)
    elif isinstance(obj, np.ndarray):
        size += obj.nbytes
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)

    return size


GB_UNIT = 1024 ** 3
# BaseDataset should contain several function like 
#   load from the disk < with a list of target data
class PretrainDataset(Dataset):
    ''' PretrainDataset
    Desc: 
        It primarily meets the requirements by the procedure implementation of this paper.
    
    Member Variables:
        @dataPathPrefix:            The location of the data with placement and dataset annotated.
        @transform_x, @transform_y: Transform function to be applied on the data(x) and label(y).
        @trainValSplit:             Identify the character of the dataset, following the default split in train and validation data (train:val = 8:2) in time order.
        @taskName:                  The task that labels belong to.
        @sampleTimeWindow:          The size of the time window of the sample.

        // // // // // // // // // // // // (*)special implementaion variables // // // // // // // // // // // // 
        @sampleMiniWindow:          Whether we need to split the sample in the time dimension, which will create a new dimension of frames in 200 ms.

        // // // // // // // // // // // // group control                   // // // // // // // // // // // // 
        @targetFolders:             Split the dataset for future setting implementation.

        // // // // // // // // // // // // @**kwargs                       // // // // // // // // // // // // 
        @EXCLUDE_NONE:              Exclude the None(0) label sample in the dataset on given tasks.
        @augmentations:             Dictionary with aug. name and function. returned along with the data.
        
    Member Functions:
        prepare_index():            prepare the index of the data and labels.
        prepare_sample_offset(init:bool=False):       prepare the offset of the sample if needed
        __item__(index):            return data[index], label[index], metaInfo[index]


    '''
    def __init__(self, dataPrefix, datasetName,
                 transform_x=None,    # function @transform_x to be applied on the data
                 trainValSplit=None, 
                 taskName=None, 
                 sampleTimeWindow=200,      # 
                 sampleMiniWindow=True,     # split with mini window
                 targetFolders=None,        # involved users/devices/modalities
                 verbose=0,
                 **kwargs) -> None:         # @trainValSplit flag: [None-'all', 'train', 'val'] 
        assert taskName is not None, "Error: One should specify the taskName in use."
        # 
        self.data, self.label   = [], []
        self.sampleTimeOffset   = []                    # random each epoch

        self.metaData = {} 
        
        # buffer to store the unique data
        # self._fileCache = {} # file pairs to form the pipeline. < chain to next file\

        self.fileBuffer = {}
        self.sampleBuffer = {} # store the buffer with (_sampleSource and newStart)
        self._reset()

        self.transform_x = transform_x
        self.trainValSplit = trainValSplit              # [None, 'train', 'val']
        self.POST_SPLIT  = -1 if 'POST_SPLIT' not in kwargs else kwargs["POST_SPLIT"] 
        # -1, split the data according to timestamp (default)
        # 0/1,  split the data according to the actual portion 8:2 in the dataset
        assert self.POST_SPLIT == -1 or (self.trainValSplit != 'train' and self.trainValSplit != 'val'), 'check the POST_SPLIT settings.' 
        
        self.K_SHOT = None if 'K_SHOT' not in kwargs else kwargs["K_SHOT"]

        self.taskName             = taskName
        self.sampleTimeWindow     = sampleTimeWindow
        self.sampleMiniWindow     = sampleMiniWindow    # default: True

        self.dataIndexFilePath    = os.path.join(dataPrefix, datasetName)
        # dataFile = os.path.join(dataPathPrefix, f"labelMap_{self.taskName.replace(":", "-")}_{int(self.sampleTimeWindow)}.pkl")

        # target users
        self.targetFolders  = targetFolders

        # ===== Option in kwargs     =====
        # prepare portion of data
        self.DATA_VOLUME = 1.0 if "DATA_VOLUME" not in kwargs else kwargs["DATA_VOLUME"]
        assert self.DATA_VOLUME <= 1.0

        #   EXCLUDE_NONE: exclude the label of None (0).
        self.EXCLUDE_NONE   = kwargs["EXCLUDE_NONE"]    if "EXCLUDE_NONE" in kwargs  else False
        self.augmentations  = kwargs["augmentations"]   if "augmentations" in kwargs else dict()
        self.HDF5 = kwargs["HDF5"] if "HDF5" in kwargs else False
        self.CACHE_SAMPLE    = kwargs["CACHE_SAMPLE"] if "CACHE_SAMPLE" in kwargs else False

        # for capture24 largest dataset used in the experiments.
        self.MMAP_MODE   = False if "MMAP_MODE" not in kwargs else kwargs["MMAP_MODE"]
        self.SHUFFLE     = False if "SHUFFLE" not in kwargs else kwargs["SHUFFLE"]

        # ===== static configuration ===== 
        self.verbose         = verbose
        self.BASE_TASK       = DATASET_TO_TASK[datasetName][DEFAULT_GRAINED_TASK] # static setting loaded from a configuration file.
        self.EMBEDDING_MODEL = kwargs["EMBEDDING_MODEL"] if "EMBEDDING_MODEL" in kwargs else "all-mpnet-base-v2"
        with open(os.path.join(DATA_CATEGORY_PATH, f"{self.EMBEDDING_MODEL}_label2Embedding.pkl"), "rb") as f:
            self.EMBEDDING_MODEL                     = pickle.load(f)
            self.EMBEDDING_MODEL['label2Embeddings'] = dict([(tuple(k[2:-1]), torch.tensor(v, dtype=torch.float64)) for k, v in self.EMBEDDING_MODEL['label2Embeddings'] if k[1] == datasetName])
            self.EMBEDDING_MODEL['label2Embeddings'][(self.BASE_TASK, 0)] = torch.tensor(self.EMBEDDING_MODEL["unknownEmbeddings"], dtype=torch.float64)
            del self.EMBEDDING_MODEL['prompt']
            if self.verbose:
                print(f"INFO: Dataset {datasetName} using embeddings derived from {self.EMBEDDING_MODEL['model_']}.")
            
        # prepare the map of y label.
        print(self.sampleTimeWindow)
        with open( os.path.join(self.dataIndexFilePath, f"metaInfo_{self.BASE_TASK.replace(':', '-')}_{int(self.sampleTimeWindow)}.cache"), "r") as f:
            self._BASE_TASK_yMap    = decode_from_string(f.read())["labelSet"]
            if 0 not in self._BASE_TASK_yMap:
                self._BASE_TASK_yMap = (0, *self._BASE_TASK_yMap)

        with open( os.path.join(self.dataIndexFilePath, f"metaInfo_{self.taskName.replace(':', '-')}_{int(self.sampleTimeWindow)}.cache"), "r") as f:
            self._CURRENT_TASK_yMap = decode_from_string(f.read())["labelSet"]
            if 0 not in self._CURRENT_TASK_yMap:
                self._CURRENT_TASK_yMap = (0, *self._CURRENT_TASK_yMap)
            self._sampleWeight = torch.ones((1, len(self._CURRENT_TASK_yMap))).float()

        if verbose:
            print(self._BASE_TASK_yMap, self._CURRENT_TASK_yMap)

        self._TRIM_TRAINING_DATA = False if "TRIM_TRAINING_DATA" not in kwargs else kwargs["TRIM_TRAINING_DATA"] 
        self._seed = 42 if "SEED" not in kwargs else kwargs["SEED"]
        set_seed(self._seed)
        # ================================ 
        self._prepare_index()
        self.prepare_sample_offset(init=True)


    def _reset(self):
        self.fileBuffer.clear() # buffer to store the unique data
        self.sampleBuffer.clear() # store the buffer with (_sampleSource and newStart)

    # check with transformation
    def checkBaseTask(self, i):
        if i in self._BASE_TASK_yMap:
            return self._BASE_TASK_yMap.index(i)
        return 0
    
    def checkCurTask(self, i):
        if i in self._CURRENT_TASK_yMap:
            return self._CURRENT_TASK_yMap.index(i)
        return 0
    
    
    def transform_y_core(self, y):
        return np.bincount(list(map(lambda i: self.checkBaseTask(i), y))).argmax()
    
    def prepare_sample_offset(self, init=False, segmentation=6):
        # prepare the sample offset
        # offset 
        if init:
            self.sampleTimeOffset = np.zeros((len(self.label), ))
        else:
            self.sampleTimeOffset = np.random.choice(range(0, self.sampleTimeWindow // 2 + 1, self.sampleTimeWindow // segmentation), size=len(self.label))

    def shuffleInFile(self):
        ''' shuffleInFile (Mannually)
        Shuffle the file orders to realize equivalent function to shuffle the samples.
        # self.SHUFFLE and self.MMAP_MODE
        The data is shuffled in file order (primary function).
        '''
        # (fileName, rmap[k][i])
        # fileSet = reduce(lambda re, x: re + [x[0],] if x[0] not in re else re, self.data, [])
        # fileSet = random.sample(fileSet, len(fileSet))

        grabFile2Index = {}
        
        for i, (x, _) in enumerate(self.data):
            if x not in grabFile2Index:
                grabFile2Index[x] = []
            grabFile2Index[x] += [i,]

        fileSet = random.sample(list(grabFile2Index.keys()),                         
                                len(grabFile2Index))
        
        reIndex = []
        fileIndexes = []
        for i, f in enumerate(fileSet):
            fileIndexes += grabFile2Index[f]
            if (i + 1) % 3 == 0 or i == len(fileSet) - 1:
                if self.SHUFFLE and self.MMAP_MODE:
                    reIndex += random.sample(fileIndexes, len(fileIndexes))
                else:
                    reIndex += fileIndexes
                fileIndexes = []
        
        new_data, new_label  = [], []
        for i in reIndex:
            new_data += [self.data[i]]
            new_label += [self.label[i]]
        
        self.data = new_data
        self.label = new_label

    def _prepare_index(self):

        dataFile = os.path.join(self.dataIndexFilePath, f"labelMap_{self.taskName.replace(':', '-')}_{int(self.sampleTimeWindow)}.pkl") 
        
        with open(dataFile, "rb") as f:
            labelMap = pickle.load(f)

        with open(dataFile.replace("labelMap", "metaInfo").replace(".pkl", ".cache"), "r") as f:
            metaInfo = dict([(folder, kMap) for folder, kMap, _ in decode_from_string(f.read())['stat']])

        data_tr, data_val = [], [] 
        lastFileName = None

        classResampledCnt = torch.zeros(len(self._CURRENT_TASK_yMap)) + 1e-7
        classActualCnt    = torch.zeros(len(self._CURRENT_TASK_yMap)) + 1e-7 # lb
        for folder, rmap, label, timeStmp in labelMap: 
            # exclude other folders with filter function or list 
            if self.targetFolders: 
                if callable(self.targetFolders) and not self.targetFolders(folder): 
                    continue 
                elif not callable(self.targetFolders) and folder not in self.targetFolders: 
                    continue 
            rmap        = dict(rmap)
            label       = dict(label)
            timeStmp    = dict(timeStmp)
            
            for k in rmap: 
                fileName = os.path.join(folder, k) 
                
                # ########### update the file caches. #########
                # if lastFileName is None:
                #     lastFileName = fileName
                    
                # if fileName != lastFileName:
                #     self._fileCache[lastFileName] = fileName
                #     lastFileName = fileName
                # #############################################

                resampled_ind_ = {}
                # TODO: resample the data.
                if self._TRIM_TRAINING_DATA:
                    curMetaInfo     = dict(metaInfo[folder])[k]
                    if None in curMetaInfo: # merge key None.
                        if 0 in curMetaInfo:
                            curMetaInfo[0] += curMetaInfo[None]
                        else:
                            curMetaInfo[0] = curMetaInfo[None]

                        del curMetaInfo[None]
                    labelCnt        = [curMetaInfo[j] if j in curMetaInfo else 0 for j in range(max(list(curMetaInfo.keys())))]
                    resampledNum    = max(np.median(list(filter(lambda x: x!=0, labelCnt))), 
                                        self._TRIM_TRAINING_DATA) # used 3 hours data

                    for c, num in enumerate(labelCnt):
                        if num <= resampledNum:
                            continue
                        if num == 0: # skip the empty class
                            continue
                        num_ = np.array(sorted(random.sample(range(int(num * 0.7)), int(resampledNum * 0.7))))
                        num_ = np.concatenate([[num_[0] + 1,], num_[1:] - num_[:-1]])
                        resampled_ind_[c] = num_.tolist()

                trainValDivider = timeStmp[k] 
                for i in range(len(rmap[k])): 
                    c = label[k][i] 
                    
                    if self.EXCLUDE_NONE and c == 0: 
                        continue 

                    # # calculate the number of the samples from the target class
                    # classActualCnt[self.checkBaseTask(c)] += num
                    # if num <= resampledNum:
                    #     classResampledCnt[self.checkBaseTask(c)] += num
                    #     continue
                    # classResampledCnt[self.checkBaseTask(c)] += resampledNum
                    cur_pos = self.checkCurTask(c)

                    if c in trainValDivider and i < trainValDivider[c]: 
                        if self.trainValSplit == 'train':
                            classActualCnt[cur_pos] += 1
                    elif c != 0:
                        if self.trainValSplit == 'val':
                            classActualCnt[cur_pos] += 1
                    elif self.trainValSplit == 'train':
                        classActualCnt[cur_pos] += 1

                        # if self.trainValSplit == 'val' and c in trainValDivider and i >= trainValDivider[c]:
                        #     classActualCnt[cur_pos] += 1
                        # elif self.trainValSplit == 'train' and c in trainValDivider and i < trainValDivider[c]:
                        #     classActualCnt[cur_pos] += 1
                        
                    if c in resampled_ind_: 
                        # skip if self._TRIM_TRAINING_DATA and labels exceed the target data.
                        if len(resampled_ind_[c]) == 0:
                            continue

                        resampled_ind_[c][0] -= 1
                        if resampled_ind_[c][0] == 0:
                            resampled_ind_[c].pop(0)
                        else:
                            continue
                        
                    if c in trainValDivider and i < trainValDivider[c]: 
                        data_tr +=  [((fileName, rmap[k][i]), c)] 
                        if self.trainValSplit == 'train':
                            classResampledCnt[cur_pos] += 1
                    elif c != 0: 
                        data_val += [((fileName, rmap[k][i]), c)] 
                        if self.trainValSplit == 'val':
                            classResampledCnt[cur_pos] += 1
                    else: 
                        data_tr +=  [((fileName, rmap[k][i]), c)] # 0 for only training  
                        if self.trainValSplit == 'train':
                            classResampledCnt[cur_pos] += 1
        
        self._sampleWeight = classActualCnt /  classResampledCnt
        print(f"Mean sample weights: {self._sampleWeight.mean()}")
        print(f"Max  sample weights: {self._sampleWeight.max()}")
        print(f"Min  sample weights: {self._sampleWeight.min()}")
        print(f"{len(data_tr)}, {len(data_val)}, {len(self._CURRENT_TASK_yMap)}")

        if len(data_tr) == 0:
            self.data, self.label = [], []

        elif self.trainValSplit == 'train':  
            self.data, self.label = zip(*data_tr)

        elif self.trainValSplit == 'val':
            if not (len(self._CURRENT_TASK_yMap) == 1 and 0 in self._CURRENT_TASK_yMap):
                self.data, self.label = zip(*data_val)
            else:
                self.data, self.label = [], []
                
        else:
            self.data, self.label = zip(*(data_tr + data_val))

        # prepare new dataset following the sample order.
        if self.DATA_VOLUME != 1.0:
            def trimByOrder(data, label, portion):
                rst = {}
                label_set = set(label)
                for i in range(len(data)):
                    f_ = self.data[i][0]
                    if f_ not in rst:
                        rst[f_] = {
                            k: 0 for k in label_set
                        }
                    rst[f_][label[i]] += 1

                accumulated_loss=0.0
                for f_ in rst:
                    for k in rst[f_]:
                        if rst[f_][k] == 0:
                            continue
                        tmp_ = rst[f_][k] * portion + accumulated_loss 
                        rst[f_][k] = min(max(1, int(tmp_)), rst[f_][k])
                        # print(tmp_, accumulated_loss)
                        accumulated_loss = tmp_ - rst[f_][k]
                    
                return rst

            labelStat = trimByOrder(self.data, self.label, self.DATA_VOLUME)
            picked_data_label = []
            ordered_data = sorted(zip(self.data, self.label))
            labelStat_ = copy.deepcopy(labelStat)
            for d_, l_ in ordered_data:
                f_ = d_[0]
                cnt = labelStat_[f_][l_]
                if cnt == 0:
                    continue
                picked_data_label.append((d_, l_))
                labelStat_[f_][l_] -= 1
            self.data, self.label = zip(*picked_data_label)

        # POST_SPLIT
        if self.POST_SPLIT != -1:
            X_train, X_test, y_train, y_test = train_test_split(self.data, self.label, test_size=0.2, random_state=65536)
            print(X_train[:5], y_train[:5], file=sys.stderr)
            print(X_test[:5], y_test[:5], file=sys.stderr)

            if self.POST_SPLIT == 0: # 'train'
                self.data, self.label = X_train, y_train                
            elif self.POST_SPLIT == 1:
                self.data, self.label = X_test, y_test


        # for few-shot settings: train and validation data remains the same.
        if self.K_SHOT is not None: # 1, 2, 3, 5, 10
            assert self.trainValSplit is None and self.POST_SPLIT == -1, "Make sure the correct configuration of the data."
            kshot, sample_seed = self.K_SHOT
            new_data, new_label = [], []

            local_np_random = np.random.RandomState(sample_seed) # each fold could be different

            label_cnt = {l: [] for l in self._CURRENT_TASK_yMap}
            for i, l in enumerate(self.label):
                assert l in label_cnt
                label_cnt[l] += [i]
            
            label_cnt_ = {l: local_np_random.choice(label_cnt[l], (kshot,), replace=False) if len(label_cnt[l]) >= kshot else [] for l in label_cnt if l != 0 and len(label_cnt[l]) != 0}
            trimmed_index = sorted(np.concatenate(list(label_cnt_.values())))
            label_desc = {l: len(label_cnt[l]) for l in label_cnt}
            assert len(trimmed_index) == kshot * len(label_cnt_), f"{label_desc}"
            print(trimmed_index[:10])
            
            for i in trimmed_index:
                new_data    += [ self.data[i]]
                new_label   += [self.label[i]]
            self.data, self.label = new_data, new_label


        if not self.MMAP_MODE:
        # pre_init load all of the data into the memory in advance.
            for curFileName, _ in self.data:
                if curFileName not in self.fileBuffer:
                    self.grabDataFile(curFileName)


    def load_file_data_hdf5(self, fileName):
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        if self.verbose:
            print(f"[INFO] loading {fileName.replace('.npz', '.hdf5')}", file=sys.stderr)

        data = {}
        hdf_file = h5py.File(os.path.join(self.dataIndexFilePath, fileName.replace(".npz", ".hdf5")), "r")
        
        data["rawData"] =   hdf_file["rawData"][:]
        data["timeStamp"] = hdf_file["timeStamp"][:]
        data["labelSeq"]  = hdf_file["labelSeq"][:]
        timeSlice = {
            200: hdf_file[f"timeSlice-{200}"][:],
        }
        if self.sampleTimeWindow not in timeSlice:
            timeSlice[self.sampleTimeWindow] = hdf_file[f"timeSlice-{int(self.sampleTimeWindow)}"][:]
        data["timeSlice"] = timeSlice

        data["used"] = 0
        # data["hdf_file"] = hdf_file
        hdf_file.close()
        if self.verbose:
            print(f"[INFO] {fileName.replace('.npz', '.hdf5')} loaded", file=sys.stderr)
        return data
        # 
        # rawData     = np.array(hdf_file["rawData"])
        # timeStamp   = np.array(hdf_file["timeStamp"])
        # labelSeq    = np.array(hdf_file["labelSeq"])
        # timeSlice = {}
        # for interval in [200, 800, 1000, 2000, 3000, 5000, 6000]:
        #     timeSlice[interval]  = np.array(hdf_file[f"timeSlice-{interval}"])
    def load_file_data(self, fileName):
        # 
        temp = np.load(os.path.join(self.dataIndexFilePath, fileName), allow_pickle=True)
        if self.verbose:
            print(f"[INFO] loading {fileName}", file=sys.stderr)

        data = dict(temp)
        data["timeSlice"] = data["timeSlice"].item()
        data["used"] = 0

        temp.close()
        if self.verbose:
            print(f"[INFO] {fileName} loaded", file=sys.stderr)
        return data

    # async def delDataFile(self, buffer_keys):
    #     for buffer_key in buffer_keys:
    #         del self.fileBuffer[buffer_key]
    #     print(f"[INFO] Reset fileBuffer DONE.", file=sys.stderr)

    def grabDataFile(self, fileName):
        
        if fileName not in self.fileBuffer:
            if self.MMAP_MODE:
                # self.fileBuffer[curFileName] = np.load(os.path.join(self.dataIndexFilePath, curFileName), allow_pickle=True, mmap_mode='r')
                if len(self.fileBuffer) >= 3:
                    # print(f"[INFO] Updating fileBuffer: {get_size(self.fileBuffer)} GB, sample buffer {get_size(self.sampleBuffer) / (1024 ** 3)} GB", file=sys.stderr)
                    if self.verbose:
                        print(f"[INFO] Reseting fileBuffer.", file=sys.stderr)
                    # fileBuffer_ = sorted([(self.fileBuffer[k]["used"]/len(self.fileBuffer[k]["timeSlice"][self.sampleTimeWindow]), k) for k in self.fileBuffer], reverse=True)
                    # prune top two data
                    # if self.HDF5:
                    #     for k in self.fileBuffer:
                    #         self.fileBuffer[k]["hdf_file"].close()
                        # self.fileBuffer[fileBuffer_[0][1]]["hdf_file"].close()

                    # del self.fileBuffer[fileBuffer_[0][1]]
                    self.fileBuffer.clear() # reset all file.


            # print(f"[INFO] task {fileName} loaded from disk.", file=sys.stderr)
            if self.HDF5:
                self.fileBuffer[fileName] = self.load_file_data_hdf5(fileName)
            else:
                self.fileBuffer[fileName] = self.load_file_data(fileName)
        
        # return self.fileBuffer[fileName]

    def __getitem__(self, index):
        
        if self.verbose:
            t_1 = time.time()

        curFileName, ind = self.data[index]

        if curFileName not in self.fileBuffer:
            self.grabDataFile(curFileName)

        offset = self.sampleTimeOffset[index]

        curTimeStamp = self.fileBuffer[curFileName]["timeStamp"]

        start, end_1, end_2   = self.fileBuffer[curFileName]["timeSlice"][self.sampleTimeWindow][ind]
        if self.HDF5:
            start, end_1, end_2   = int(start), int(end_1), int(end_2)
        timeStamp             = curTimeStamp[start:end_2]

        
        if self.verbose:
            t_2 = time.time()

        newStart, newEnd = start, end_1

        # dense sampling
        if offset > 0:
            temp = [(i - timeStamp[0]) >= offset for i in timeStamp]
            if True in temp:
                newStart   = temp.index(True)
                newEnd     = [(i - timeStamp[newStart]) >= self.sampleTimeWindow for i in timeStamp]

                if True in newEnd:
                    newEnd = newEnd.index(True)
                else:
                    newEnd = end_2 - start

                if abs((timeStamp[newEnd - 1] - timeStamp[newStart]) - (timeStamp[end_1 - start - 1] - timeStamp[0])) \
                    <= self.sampleTimeWindow / (end_1 - start):
                    newStart  += start
                    newEnd    += start
                else:
                    newStart, newEnd = start, end_1
        else:
            newStart, newEnd = start, end_1
        
        key = (curFileName, ind, newStart)
        if self.verbose:
            if key in self.sampleBuffer:
                print(f"Cache hit for {key}")
        # else:
        #     print(f"Cache miss for {key}")
        
        if self.verbose:
            t_3 = time.time()

        # update from the cache
        if key in self.sampleBuffer:
            return self.sampleBuffer[key]
        
        # self.fileBuffer[curFileName]["used"] += 1

        # prepare the returned sample
        timeStamp   = curTimeStamp[newStart:newEnd]
        rawData     = self.fileBuffer[curFileName]["rawData"  ][newStart:newEnd]
        labelSeq    = self.fileBuffer[curFileName]["labelSeq" ][newStart:newEnd] # prepare the data in small window
        
        
        if self.verbose:
            t_4 = time.time()

        if self.sampleMiniWindow:
            miniWindowSize = 200 # 200 ms / frame
            accumulatedWindowSize = miniWindowSize
            sigEnds = []
            for i, ts in enumerate(timeStamp):
                if ts - timeStamp[0] > accumulatedWindowSize:
                    accumulatedWindowSize += miniWindowSize
                    sigEnds += [i]
            rDataSplit = np.split(rawData, sigEnds)[:self.sampleTimeWindow//miniWindowSize]
            rTimeStampSplit = np.split(timeStamp, sigEnds)[:self.sampleTimeWindow//miniWindowSize]

            transX = []
            for t, d in zip(rTimeStampSplit, rDataSplit):
                transX += [torch.as_tensor(self.transform_x(t, d))]

            rawData    = torch.stack(transX).float() # F, L, C
            # Split and stack the sample in frames:
            # - ensure the transform_x to function well on the rawData.
            labelSeq   = torch.as_tensor(list(map(lambda l: self.transform_y_core(l), np.split(labelSeq, sigEnds)))).long() # F,
        else:
            rawData    = torch.as_tensor(self.transform_x(timeStamp, rawData)).float()
            labelSeq   = torch.as_tensor([self.transform_y_core(labelSeq),]).long() # 1,
        
        if self.verbose:
            t_5 = time.time()
            print(f"[INFO] time used: fRead1 {t_2-t_1 : .5f}, dense sample {t_3-t_2: .5f}, fRead2 {t_4 - t_3: .5f}, transform {t_5 - t_4:.5f}.")

        # rawData = torch.as_tensor(rawData).float()
        label =   torch.as_tensor(self._CURRENT_TASK_yMap.index(self.label[index])).long()
        metaInfo = {}
        
        if self.sampleMiniWindow:
            # metaInfo["labelSeq"]     = torch.as_tensor(labelSeq, dtype=torch.long) # label sequences
            metaInfo["labelSeq"] = labelSeq
            metaInfo["embeddingSeq"] = torch.stack(list(map(lambda i: self.EMBEDDING_MODEL['label2Embeddings'][(self.BASE_TASK, self._BASE_TASK_yMap[i])]
                                                    , labelSeq))) # F, embedding shape
            

        if len(self.augmentations) != 0:
            augData = {k: self.augmentations[k](rawData) for k in self.augmentations}
            metaInfo.update(augData)

        metaInfo["_sampleSource"] = (curFileName, ind) # left for future debugging. newStart

        # if get_size(self.sampleBuffer) > 3.0 * GB_UNIT:
        #     print("[INFO] clear cache.", file=sys.stderr)
        #     self.sampleBuffer.clear()
        if self.CACHE_SAMPLE:
            self.sampleBuffer[key] = (rawData.transpose(-1, -2), label, metaInfo)
            
        return rawData.transpose(-1, -2), label, metaInfo
        # Make sure the feature is of the size (F, C, L) 
        # return self.sampleBuffer[key]

    def __len__(self):
        return len(self.label)

class DatasetFactory(object):
    # DatasetFactory derives BaseDataset object in need.
    def __init__(self, dataRootPath, tarPlacement="WRIST", tarDataset="all", **kargs) -> None:

        self.tarDataset = [tarDataset, ]
        if tarDataset == 'all':
            self.tarDataset = [i for i in os.listdir(os.path.join(dataRootPath, tarPlacement)) if "exclude" not in kargs or i not in kargs["exclude"]]

        self.allDataInRecord = {
            d: list(filter(lambda x: os.path.isdir(os.path.join(dataRootPath, tarPlacement, d, x)), os.listdir(os.path.join(dataRootPath, tarPlacement, d)))) 
            for d in self.tarDataset
        }

    def show_option(self, verbose=1, **kargs) -> dict:
        ''' show_option
        params:
            @verbose: 1 - show details, 0 - show brief summary.
            @**kargs: leftover.
        return:
            @config: meta information of the datasets involved 
        '''
        # prepare the metaInfo which includes,
        #   1. users involved.
        #   2. devices and modality involved.
        #   3. special cases involved.
        uniqueWithReduce = lambda tarList: reduce(lambda re, x: re + [x] if x not in re else re, tarList, [])

        users = {d: uniqueWithReduce([i.split("+")[0] for i in self.allDataInRecord[d]]) for d in self.allDataInRecord}

        devsMods = []
        for d in self.allDataInRecord:
            merged = {}
            curDevMods = [(d, *i) for i in uniqueWithReduce([(i.split("+")[1] + "+" + i.split("+")[-1], i.split("+")[-2])   for i in self.allDataInRecord[d]])]
            for d, dev, mod in curDevMods:
                key_ = dev
                if key_ not in merged:
                    merged[key_] = []
                merged[key_] += [mod]
            devsMods += [(d, merged)]

        # tasks involved
        tarTasks = {d: DATASET_TO_TASK[d] for d in self.allDataInRecord}
        defaultTask = {d: DATASET_TO_TASK[d][DEFAULT_GRAINED_TASK] for d in self.allDataInRecord}

        if verbose:
            print("==========      user      ==========")
            pprint(users, width=100, compact=True)
            print("==========    devsMods    ==========")
            pprint(dict(devsMods))
            print("==========  task Involved ==========")
            pprint(tarTasks, width=100, compact=True)
            print("==========  default task  ==========") # it is important for later implementation.
            pprint(defaultTask)
            print("==========    sampleTW    ==========")
            pprint([200, 800, 1000, 2000, 3000, 5000, 6000])
            print("==========    print end   ==========")
            
        return {
            "users": users, 
            "devsMods": dict(devsMods),
            "tarTasks": tarTasks,
            "sampleTimeWindow": [200, 800, 1000, 2000, 3000, 5000, 6000]
        }
    # calls
    # single modality
    #   this.prepare_dataset(tarDataset ='2012-OPPORTUNITY', 
    #                        taskName   ='annotation', 
    #                        tarModality='acc', 
    #                        transform_x=lambda x: resample(x, 20), 
    #                        sampleTimeWindow=200, 
    #                        verbose=0, 
    #                        # pass through the @kwargs 
    #                        tarUserset = ["S1", "S2", "S3"], 
    #                        augmentations = {"randpermute": randpermute, }, 
    #                        trainValSplit = 'train' 
    # )
    def prepare_dataset(self, datasets, taskName, tarModality, 
                                transform_x, sampleTimeWindow=200, 
                                verbose=0, **kwargs):
        ''' prepare_dataset
        params:
            @datasets(str):          prepare the datasets
            @tarModality(str):       prepare the modality, options = ["acc", "linearAcc", "gyro", "magnetic", ...]
            @taskName(str):          return the tasks in use.
            @transform_x:       the function transforms the data.
            @sampleTimeWindow:  the time window selected.
            @verbose:           1 - show details, 0 - show brief summary.
            @kargs:
                tarUserSet   = [xx, xx, xx, ...]:    for train/test split.
                augmentations:  dictionary of augmentation functions, each one entry will raise a new entry in the metaInfo returned from the Dataset.
        return:
            @dataset:           a BaseDataset that returns data, label and metaInfo
        '''
        pass


    
class DataPrefetcherThread:
    def __init__(self, loader, device, num_prefetched_batches=25, verbose=0):
        self.loaderit = None
        self.loader = loader
        self.device = device
        self.num_prefetched_batches = num_prefetched_batches
        self.prefetched_batches = deque()
        self.loader_lock = threading.Lock()
        self.stream = torch.cuda.Stream()
        self.running = True
        self.prefetch_thread = None
        self.verbose= verbose

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
            if self.verbose:
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
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            self.prefetch_thread.join()


class InfiniteDataLoader(IterableDataset): # used in the pretraining backbone model.
    # USAGE: infinite_dloader = InfiniteDataLoader(dataLoader, shuffle_manually=True) 
    def __init__(self, dataLoader, shuffle_manually:bool = False, segmentation = 15, use_prefetcher = False, device = None) -> None:
        super(InfiniteDataLoader, self).__init__()
        self.dataLoader = dataLoader
        self.shuffle_manually = shuffle_manually    # only work for dataset with MMAP_MODE
        self.segmentation = segmentation            # used for dense sampling
        self.iteration_cnt = 0
        self.use_prefetcher = use_prefetcher
        self.device = device

    def __iter__(self): # endless iteration
        while True:
            if self.shuffle_manually and self.dataLoader.dataset.MMAP_MODE:
                self.dataLoader.dataset.shuffleInFile() 

            dataLoader = DataPrefetcherThread(self.dataLoader, self.device) if self.use_prefetcher else self.dataLoader
            iterator = iter(dataLoader)    
            for _ in range(len(dataLoader)):
                yield next(iterator)

            self.iteration_cnt += 1
            print(f"[INFO] {self.dataLoader.dataset.dataIndexFilePath} runs out, current iteration: {self.iteration_cnt}", file=sys.stderr) # update the iterations state.

            if self.segmentation:
                self.dataLoader.dataset.prepare_sample_offset(init=False, segmentation=self.segmentation) # dense sampling
            
            iterator = iter(self.dataLoader)
            
