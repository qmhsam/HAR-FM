from functools import reduce
import dill as pickle
import os
from .serialization import decode_from_string


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# static config
DEFAULT_GRAINED_TASK = 0 # The first task as default setting
DATASET_TO_TASK = {
    "2012-OPPORTUNITY": ["annotation",      #
                         "Locomotion", 
                         "ML_Both_Arms"],
    "2012-PAMAP2":  ["activityID"],         #
    "2012-realdisp":  ["annotation",        #
                       "lowerExtremes", 
                       "other", 
                       "trunk", 
                       "upperExtremes", 
                       "wholeBody"],
    "2014-mHealth": ["activityID"],         #
    "2014-shoaib":  ["activityID"],         #
    "2015-HHAR":    ["gt"],                 #
    "2019-WISDM":   ["annotation",          #
                     "hand-oriented-eating", 
                     "hand-oriented-general", 
                     "non-hand-oriented"],
    "2020-capture24": ["annotation",        #
                       "label:Doherty2018", 
                       "label:DohertySpecific2018", 
                       "label:Walmsley2020", 
                       "label:Willetts2018", 
                       "label:WillettsMET2018", 
                       "label:WillettsSpecific2018"],
    "2020-GOTOV":   ["label"]
}

DATA_CATEGORY_PATH = "../../dataCenter"

# DATASET_TO_FOLDS = {}

# # init
# with open(f"{DATA_CATEGORY_PATH}/metaInfo_userSplit.pkl", "rb") as f:
#     userSplit = pickle.load(f)
#     DATASET_TO_FOLDS = {k: {i: j[:2] for i, j in enumerate(userSplit[k][0])} for k in userSplit}
#     del userSplit

# load task
def grabTasklabels(placement, dataset, interval, taskName = None, DATA_CATEGORY_PATH=DATA_CATEGORY_PATH):
    if dataset not in DATASET_TO_TASK:
        return None
    
    if taskName is None:
        taskName = DATASET_TO_TASK[dataset][DEFAULT_GRAINED_TASK]

    path_ = os.path.join(DATA_CATEGORY_PATH, placement, dataset, f"metaInfo_{taskName.replace(':', '-')}_{interval}.cache")

    if not os.path.exists(path_):
        return None
    
    with open(path_, "r") as f:
        TASK = decode_from_string(f.read())["labelSet"]
        if 0 not in TASK:
            TASK = (0, *TASK)

    return TASK

class TarFolderFilter:
    def __init__(self, user=None, dev=None, modality=None) -> None:
        self.user = user
        self.dev = dev
        self.modality = modality
    
    def __call__(self, fileName):
        ''' support the fileName in format: {user}+{dev}+{modality}+{option}
        '''
        flag = True
        if flag and self.user is not None:
            flag = reduce(lambda re, x: re or x == fileName.split("+")[-4], self.user, False)
        if flag and self.dev is not None:
            flag = reduce(lambda re, x: re or x  == fileName.split("+")[-3], self.dev, False)
        if flag and self.modality is not None:
            flag = reduce(lambda re, x: re or x  == fileName.split("+")[-2], self.modality, False)
        return flag
        