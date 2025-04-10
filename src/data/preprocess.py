import torch
import random
import numpy as np
import pandas as pd
from typing import Tuple, Any
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
    

def label_encode(df: pd.DataFrame, is_single_label: bool) -> Tuple[pd.Series, np.ndarray, Any]:
    """Encode the labels

    Args:
        df (pd.DataFrame): Data frame contains labels and stratified fold index
        is_single_label (bool): If the input label is single

    Returns:
        Tuple[pd.Series, np.ndarray, Any]: Stratified fold index, encoded label, and encoder model
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
            
    return df.strat_fold, y, encoder


def data_split(data: np.ndarray, fold: pd.Series, y: np.ndarray) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train, validation, test sets with provided stratified folds

    Args:
        data (np.ndarray): ECG signal records
        fold (pd.Series): Stratified fold index
        y (np.ndarray): Record labels

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        Train data, train labels, validation data, validation labels, test data, test labels
    """
    lower = min(fold)
    upper = max(fold)
    val_fold = random.randint(lower, upper)
    test_folds = random.sample([num for num in range(lower, upper + 1) if num != val_fold], 2)
    train_folds = [num for num in range(lower, upper + 1) if num != val_fold and num not in test_folds]
    train_indexes = np.where(fold.isin(train_folds))[0]
    val_indexes = np.where(fold == val_fold)[0]
    test_indexes = np.where(fold.isin(test_folds))[0]
    train_X, train_y = data[train_indexes], y[train_indexes]
    val_X, val_y = data[val_indexes], y[val_indexes]
    test_X, test_y = data[test_indexes], y[test_indexes]
    return train_X, train_y, val_X, val_y, test_X, test_y


def data_standardize(train_X: np.ndarray, val_X: np.ndarray, test_X: np.ndarray) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Standardize the records data base on scaler model fitted on the train records

    Args:
        train_X (np.ndarray): Train records to fit the model and standardized
        val_X (np.ndarray): Validation records to be standardized
        test_X (np.ndarray): Test records to be standardized

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, Any]: Standardized train, validation, test records, 
        and the fitted scaler model
    """
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X.reshape(train_X.shape[0], -1)).reshape(train_X.shape)
    val_X = scaler.transform(val_X.reshape(val_X.shape[0], -1)).reshape(val_X.shape)
    test_X = scaler.transform(test_X.reshape(test_X.shape[0], -1)).reshape(test_X.shape)
    return train_X, val_X, test_X, scaler


def get_dataloader(train_X: np.ndarray, train_y: np.ndarray, val_X: np.ndarray, val_y: np.ndarray, 
                   test_X: np.ndarray, test_y: np.ndarray, batch_size: int = 256, seed: int = 42) \
    -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Put records and labels into DataLoaders for training, validation, and testing

    Args:
        train_X (np.ndarray): Train records
        train_y (np.ndarray): Train record labels
        val_X (np.ndarray): Validation records
        val_y (np.ndarray): Validation record labels
        test_X (np.ndarray): Test records
        test_y (np.ndarray): Test record labels
        batch_size (int, optional): Batch size for training, validating, and testing. Defaults to 256.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, test DataLoaders
    """
    # Convert data and labels into torch tensors
    train_X = torch.tensor(train_X.transpose(0, 2, 1), dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float16)
    val_X = torch.tensor(val_X.transpose(0, 2, 1), dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float16)
    test_X = torch.tensor(test_X.transpose(0, 2, 1), dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float16)
    
    # Generate dataloaders
    g = torch.Generator()
    g.manual_seed(seed) # Set seed to ensure the reproducibility for train loader
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def main():
    pass


if __name__ == "__main__":
    main()
