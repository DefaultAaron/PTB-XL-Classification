import numpy as np
import pandas as pd
from typing import Tuple
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder

def data_clean(data: np.ndarray, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Remove the records with no label
    Args:
        data (np.ndarray): ECG signal records
        df (pd.DataFrame): Record labels

    Returns:
        Tuple[np.ndarray, pd.DataFrame]: Records with at least one label
    """
    new_data = []
    new_df = []
    for i, (_, row) in enumerate(df.iterrows()):
        labels = row['diagnostic']
        signal = data[i]
        if len(labels) > 0:
            new_df.append(row)
            new_data.append(signal)
        
    data = np.array(new_data)
    df = pd.DataFrame(new_df)
    return data, df
    

def label_encode(df: pd.DataFrame, is_single_label: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Encode the labels

    Args:
        df (pd.DataFrame): Data frame contains labels and stratified fold index
        is_single_label (bool): If the input label is single

    Returns:
        Tuple[np.ndarray, np.ndarray]: Stratified fold index and encoded label
    """
    if is_single_label:
        encoder = LabelEncoder()    
    else:
        encoder = MultiLabelBinarizer()
    
    y = encoder.fit_transform(df.diagnostic.values)
    if not is_single_label:
        row_sums = np.sum(y, axis=1)
        if np.all(row_sums == 1):
            y = np.argmax(y, axis=1)
            
    return df.strat_fold.values, y


def data_split():
    ... # TODO Data split


def data_standardize():
    ... # TODO Data standardize


def get_dataloader():
    ... # TODO Get dataloader


def main():
    pass


if __name__ == "__main__":
    main()
