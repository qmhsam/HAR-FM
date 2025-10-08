import os
from datetime import datetime
import pandas as pd
import numpy as np

def get_latest_log(log_directory):
    # List all files in the directory
    files = os.listdir(log_directory)

    # Filter for files that match the naming pattern of your logs
    log_files = [f for f in files if f.endswith('.csv')]

    if not log_files:
        return None  # No matching log files found

    # Parse timestamps and create (filename, timestamp) pairs
    parsed_files = []
    for file in log_files:
        try:
            # Extract the timestamp part from the filename
            timestamp_str = file.split('_', 2)[-1].rsplit('.', 1)[0]
            # Parse the timestamp
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            parsed_files.append((file, timestamp))
        except (ValueError, IndexError):
            # Skip files that don't match the expected format
            print(f"Skipping file with unexpected format: {file}")

    if not parsed_files:
        return None  # No files with valid timestamps found

    # Find the file with the latest timestamp
    latest_file, _ = max(parsed_files, key=lambda x: x[1])

    # Full path to the latest log file
    return os.path.join(log_directory, latest_file)


results_parent = os.path.join(".", "results_w1000", 
                              "postTrain_finetuneAll", "WRIST")

print(results_parent)
dataset2Rst = {}
for f in os.listdir(results_parent):
    for mod_ in ["acc", "gyro", "magnetic"]:
        for fold_ in range(5):
            tarFold = os.path.join(results_parent, f, mod_, str(fold_))
            if not os.path.exists(f"{tarFold}/test-logs"):
                continue
            tarLog = get_latest_log(f"{tarFold}/test-logs")
            # print(tarLog)
            curKey = (f, mod_)
            if f not in dataset2Rst:
                dataset2Rst[f] = {}
            if mod_ not in dataset2Rst[f]:
                dataset2Rst[f][mod_] = []
            # print()
            dataset2Rst[f][mod_] += [pd.read_csv(tarLog, index_col=["user-dev-option",]).loc["Overall", "Accuracy"]]
            
            
def apply_(x):
    if not isinstance(x, list) and pd.isna(x):
        return x
    else:
        return np.mean(x)
    
target_groups = ["2012-PAMAP2", "2014-mHealth", "2014-shoaib", "2015-HHAR", "2020-GOTOV"]
target_groups = [i for i in target_groups if i in dataset2Rst.keys()]
print(pd.DataFrame(dataset2Rst).map(apply_).T.loc[target_groups, :].sort_index())