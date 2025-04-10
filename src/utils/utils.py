import os
import torch
import random
import pickle
import sklearn
import numpy as np
from typing import Any
from matplotlib.pyplot import savefig

ROOT = os.getcwd()
while not os.path.isfile(f'{ROOT}/README.md'):
    ROOT = os.path.dirname(ROOT)
ROOT += "/"
 
MODEL_PATH = ROOT + "models/"
RAW_DATA_PATH = ROOT + "data/raw/"
PROCESSED_DATA_PATH = ROOT + "data/processed/"
FIGURE_PATH = ROOT + "results/figures/"
METRICS_PATH = ROOT + "results/metrics/"

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

def save_model(model: Any, name: str) -> None:
    """Save model with given name

    Args:
        model (Any): Model to save
        name (str): Name of the file to save the model
    """
    path = f"{MODEL_PATH}{name}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def save_figure(fig, name, **kwargs) -> None:
    # TODO Write documentation
    path = f"{FIGURE_PATH}{name}.png"
    fig.savefig(path, **kwargs)
    print(f"Figure saved to {path}")

def save_metrics():
    ... # TODO Save metrics


def main():
    print(MODEL_PATH)
    print(RAW_DATA_PATH)
    print(PROCESSED_DATA_PATH)
    print(FIGURE_PATH)
    print(METRICS_PATH)
    
if __name__ == "__main__":
    main()
