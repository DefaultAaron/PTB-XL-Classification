import os
os.environ['PYTHONHASHSEED'] = '42' # Set seed for multiple processes

from typing import Literal, Tuple
import pandas as pd
import numpy as np
import wfdb
import ast
from tqdm import tqdm
import torch
import random
import sklearn
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import pickle
from torch.utils.data import TensorDataset, DataLoader

def set_seed(seed: int = 42, determinism: bool = False) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed (int, optional): Random seed. Defaults to 42.
        determinism (bool, optional): Wether to set cudnn determinism or not, 
            set to be True may slow down the performance. Defaults to False.
    """
    np.random.seed(seed)
    random.seed(seed)
    sklearn.utils.check_random_state(seed)
    torch.manual_seed(seed)
    
    # Set seeds for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Additional PyTorch settings for cudnn determinism
        if determinism:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # Set seed for MPS
    elif torch.mps.is_available():
        torch.mps.manual_seed(seed)


def load_data(sampling_rate: Literal[100, 500], num_class: Literal[2, 5, 23, 44], path: str = 'ptb_xl/') \
    -> Tuple[np.ndarray, pd.DataFrame]:
    """Load ECG data and labels.

    Args:
        sampling_rate (Literal[100, 500]): Sampling frequency of 100Hz or 500Hz.
        num_class (Literal[2, 5, 23, 44]): Number of classes, could be either binary or multiple(5, 23, 44).
        path (str, optional): Path to the database. Defaults to ''.

    Returns:
        Tuple[np.ndarray, pd.DataFrame]: The ECG signal data be converted into an numpy array,
            and corresponding labels with stratified folds store in the pandas dataframe.
    """
    # load and convert annotation data
    df = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Aggregate the diagonstic with the record table
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic:
            if key in agg_df.index:
                if num_class == 2:
                    tmp.append('NORMAL' if key == 'NORM' else 'ABNORMAL')
                elif num_class == 5:
                    tmp.append(agg_df.loc[key].diagnostic_class)
                elif num_class == 23:
                    tmp.append(agg_df.loc[key].diagnostic_subclass)
                else:
                    tmp.append(key)
        return list(set(tmp))
    df['diagnostic'] = df.scp_codes.apply(aggregate_diagnostic)

    # Load signal data
    try:
        # Try to read data from cached file
        data = np.load(f'{path}records{sampling_rate}.npy')
        
    except FileNotFoundError:
        # Processing the raw data if is not cached
        print('Reading from raw data...')
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path + f) for f in tqdm(df.filename_lr)]
        else:
            data = [wfdb.rdsamp(path + f) for f in tqdm(df.filename_hr)]
        data = np.array([signal for signal, _ in data])
        
        # Save the processed data
        np.save(f'{path}records{sampling_rate}.npy', data)
        
    return data, df[['strat_fold', 'diagnostic']]


def perprocessing(X: np.ndarray, Y: pd.DataFrame, batch_size: int = 256, seed: int = 42, path: str = 'ptb_xl/') \
    -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Perprocessing the data: remove the records with no labels; encode the labels; 
       split into train, validation, and test datasets; standardize the signal data; 
       and combined data and label into torch DataLoaders.

    Args:
        X (np.ndarray): Signal data.
        Y (pd.DataFrame): Label table with fold number.
        batch_size (int, optional): Batch size for the data loader. Defaults to 256.
        seed (int, optional): Random seed for the train set data loader. Defaults to 42.
        path (str, optional): Path to save the encoder and scaler. Defaults to ''.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Torch DataLoaders for train, validation, and test data.
    """
    # Copy the inputs to prevent modify the local data
    data = X.copy()
    df = Y.copy()
    
    # Remove records without labels
    non_empty_indices = df.diagnostic.apply(lambda x: len(x) > 0).values
    data = data[non_empty_indices]
    df = df[non_empty_indices].reset_index(drop=True)
    
    # Encode label to one-hot scheme
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df.diagnostic.values)
    
    # Split data and label into train, validation, and test sets
    val_fold = random.randint(1, 10)
    test_folds = random.sample([num for num in range(1, 11) if num != val_fold], 2)
    train_folds = [num for num in range(1, 11) if num != val_fold and num not in test_folds]
    train_indexes = np.where(df.strat_fold.isin(train_folds))
    val_indexes = np.where(df.strat_fold == val_fold)
    test_indexes = np.where(df.strat_fold.isin(test_folds))
    train_X, train_y = data[train_indexes], y[train_indexes]
    val_X, val_y = data[val_indexes], y[val_indexes]
    test_X, test_y = data[test_indexes], y[test_indexes]
    
    # Standardize signal data
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X.reshape(train_X.shape[0], -1)).reshape(train_X.shape)
    val_X = scaler.transform(val_X.reshape(val_X.shape[0], -1)).reshape(val_X.shape)
    test_X = scaler.transform(test_X.reshape(test_X.shape[0], -1)).reshape(test_X.shape)
    
    # Save the label encoder and data scaler
    with open(path + 'encoder.pkl', 'wb') as f:
        pickle.dump(mlb, f)
    with open(path + 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Convert data and labels into torch tensors
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float16)
    val_X = torch.tensor(val_X, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float16)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float16)
    
    # Generate dataloaders
    g = torch.Generator()
    g.manual_seed(seed) # Set seed to ensure the reproducibility for train loader
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def main() -> None:
    set_seed()
    train_loader, val_loader, test_loader = perprocessing(*load_data(100, 2))


if __name__ == "__main__":
    main()
