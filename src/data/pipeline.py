import sys
sys.path.append(".")

from typing import Literal, Tuple
from torch.utils.data import DataLoader
from src.utils import save_model
from src.data import (load_data, 
                      split_labels, 
                      remove_labels,
                      data_clean, 
                      label_encode, 
                      data_split, 
                      data_standardize, 
                      get_dataloader)

def data_preparation(sampling_rate: Literal[100, 500] = 100, num_class: Literal[2, 5, 23, 44] = 2, 
                     threshold: int = 100, method: Literal["split", "remove", "multi"] = "remove", 
                     batch_size: int = 256, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Entire pipeline of the data loading and preprocessing

    Args:
        sampling_rate (Literal[100, 500], optional): Sampling frequency of 100Hz or 500Hz. Defaults to 100.
        num_class (Literal[2, 5, 23, 44], optional): Number of classes, 
            could be either binary (2) or multiple(5, 23, 44). Defaults to 2.
        threshold (int, optional): Likelihood threshold to accept labels. Defaults to 100.
        method (Literal["split", "remove", "multi"], optional): Method to handle the labels,
            could be split the labels, remove the normal labels with condition, 
            or keep the multiple label. Defaults to "remove".
        batch_size (int, optional): Batch size for training, validating, and testing. Defaults to 256.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoader
    """
    data, labels = load_data(sampling_rate, num_class, threshold)
    if method == "split":
        data, labels = split_labels(data, labels)
        fold, y, encoder = label_encode(labels, True)
        save_model(encoder, f"{sampling_rate}_{num_class}_split_encoder")
    else:
        if method == "remove":
            labels = remove_labels(labels)
        data, labels = data_clean(data, labels)
        fold, y, encoder = label_encode(labels, False)
        save_model(encoder, f"{sampling_rate}_{num_class}_{method}_encoder")
    
    train_X, train_y, val_X, val_y, test_X, test_y = data_split(data, fold, y)
    train_X, val_X, test_X, scaler = data_standardize(train_X, val_X, test_X)
    save_model(scaler, f"{sampling_rate}_{num_class}_{method}_scaler")
    train_loader, val_loader, test_loader = get_dataloader(train_X, train_y, val_X, val_y, test_X, test_y, batch_size, seed)
    return train_loader, val_loader, test_loader


def main():
    pass
    
if __name__ == "__main__":
    main()