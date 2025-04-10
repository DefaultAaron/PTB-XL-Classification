import sys
sys.path.append(".")

import ast
import wfdb
import tqdm
import numpy as np
import pandas as pd
from typing import Literal, Tuple
from src.utils import RAW_DATA_PATH, PROCESSED_DATA_PATH

def load_data(sampling_rate: Literal[100, 500], num_class: Literal[2, 5, 23, 44], threshold: int = 100) \
    -> Tuple[np.ndarray, pd.DataFrame]:
    """Load ECG data and labels.

    Args:
        sampling_rate (Literal[100, 500]): Sampling frequency of 100Hz or 500Hz.
        num_class (Literal[2, 5, 23, 44]): Number of classes, could be either binary or multiple(5, 23, 44).
        threshold (int, optional): Likelihood threshold to accept labels. Defaults to 100.

    Returns:
        Tuple[np.ndarray, pd.DataFrame]: The ECG signal data be converted into an numpy array,
            and corresponding labels with stratified folds store in the pandas dataframe.
    """
    # load and convert annotation data
    df = pd.read_csv(f"{RAW_DATA_PATH}ptbxl_database.csv", index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(f"{RAW_DATA_PATH}scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Aggregate the diagnostic with the record table
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key, value in y_dic.items():
            if key in agg_df.index and value >= threshold:
                if num_class == 2:
                    tmp.append("NORMAL" if key == "NORM" else "ABNORMAL")
                elif num_class == 5:
                    tmp.append(agg_df.loc[key].diagnostic_class)
                elif num_class == 23:
                    tmp.append(agg_df.loc[key].diagnostic_subclass)
                else:
                    tmp.append(key)
        return list(set(tmp))
    
    df["diagnostic"] = df.scp_codes.apply(aggregate_diagnostic)

    # Load signal data
    try:
        # Try to read data from cached file
        data = np.load(f"{PROCESSED_DATA_PATH}records{sampling_rate}.npy")
        
    except FileNotFoundError:
        # Processing the raw data if is not cached
        print("Reading from raw data...")
        if sampling_rate == 100:
            data = [wfdb.rdsamp(RAW_DATA_PATH + f) for f in tqdm(df.filename_lr)]
        else:
            data = [wfdb.rdsamp(RAW_DATA_PATH + f) for f in tqdm(df.filename_hr)]
        data = np.array([signal for signal, _ in data])
        
        # Save the processed data
        np.save(f"{PROCESSED_DATA_PATH}records{sampling_rate}.npy", data)
        
    return data, df[["strat_fold", "diagnostic"]]


def split_labels(data: np.ndarray, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Split the multiple labels into single label by duplicate the corresponding records

    Args:
        data (np.ndarray): ECG signal records
        df (pd.DataFrame): Record labels and stratified fold number

    Returns:
        Tuple[np.ndarray, pd.DataFrame]_: ECG signal records with single corresponding labels
    """
    new_data = []
    new_df = []
    for i, (_, row) in enumerate(df.iterrows()):
        labels = row['diagnostic']
        signal = data[i]
        for label in labels:
            new_row = row.copy()
            new_row['diagnostic'] = label
            new_df.append(new_row)
            new_data.append(signal)
        
    data = np.array(new_data)
    df = pd.DataFrame(new_df)
    return data, df


def remove_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Remove the Normal label if other label occurs

    Args:
        labels (pd.DataFrame): Data frame stores the labels

    Returns:
        pd.DataFrame: Normal labels removed while other label occurs
    """
    labels.diagnostic = labels.diagnostic.apply(lambda x: [label for label in x 
                                                           if label != "NORM" and 
                                                           label != "NORMAL"] 
                                                if len(x) > 1 else x)
    return labels


def main():
    pass


if __name__ == "__main__":
    main()
