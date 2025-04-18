from numpy import ndarray
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from typing import Union, Callable, Optional, Tuple, Dict, List

Scheduler = Union[_LRScheduler, ReduceLROnPlateau]
LossFunction = Union[
    nn.Module,  # For PyTorch loss classes like nn.CrossEntropyLoss
    Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ],  # For functional losses or custom functions
]


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    
    if torch.mps.is_available():
        return "mps"
    
    return "cpu"


def train_epoch(
    device: str,
    model: Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: LossFunction,
    scheduler: Optional[Scheduler] = None,
    step_scheduler_per_batch: bool = False,
) -> Tuple[float, float]:
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for signals, labels in dataloader:
        if device != "cpu":
            signals, labels = signals.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Step scheduler if needed per batch
        if scheduler is not None and step_scheduler_per_batch:
            if not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()

        # Track statistics
        train_loss += loss.item() * signals.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss /= train_total
    train_acc = train_correct / train_total
    return train_loss, train_acc


def val_epoch(
    device: str,
    model: Module,
    dataloader: DataLoader,
    criterion: LossFunction,
) -> Tuple[float, float]:
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        # progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        for signals, labels in dataloader:
            if device != "cpu":
                signals, labels = signals.to(device), labels.to(device)

            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs, labels)

            # Track statistics
            val_loss += loss.item() * signals.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= val_total
    val_acc = val_correct / val_total
    return val_loss, val_acc


def train_model(
    device: str,
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: LossFunction,
    num_epochs: int = 50,
    early_stopping_patience: int = 10,
    scheduler: Optional[Scheduler] = None,
    step_scheduler_per_batch: bool = False,
) -> Tuple[Module, Dict[str, List[float]]]:
    # Move model to device        
    model = model.to(device)

    # Track training history
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Early stopping
    best_val_loss = float("inf")
    best_model_state = None
    early_stopping_counter = 0

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Training
        train_loss, train_acc = train_epoch(
            device,
            model,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            step_scheduler_per_batch,
        )

        # Validation
        val_loss, val_acc = val_epoch(device, model, val_loader, criterion)

        # Update scheduler after validation if needed
        if scheduler is not None and not step_scheduler_per_batch:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def plot_training_history(history, figsize: Tuple[int, int] = (12, 5)) -> Figure:
    """
    Plot training history

    Args:
        history: Dictionary containing training history
    """
    fig = plt.figure(figsize=figsize)

    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
    return fig


def test_model(device: str, model: Module, dataloader: DataLoader) -> Tuple[List[int], List[int]]:
    # Move model to device
    model = model.to(device)
    model.eval()

    true_labels = []
    predictions = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Testing")
        for signals, labels in progress_bar:
            signals, labels = signals.to(device), labels.to(device)

            # Forward pass
            outputs = model(signals)
            _, predicted = torch.max(outputs, 1)

            # Collect results
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())

    return true_labels, predictions


def get_report(labels: List[int], predictions: List[int]) -> ndarray:
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(labels, predictions))

    return confusion_matrix(labels, predictions)


def plot_confusion_matrix(cm: ndarray, num_classes: int, figsize: Tuple[int, int] = (10, 8)) -> Figure:
    fig = plt.figure(figsize=figsize)

    if num_classes <= 5:  # For smaller number of classes, use a more detailed plot
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=range(num_classes),
            yticklabels=range(num_classes),
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
    else:  # For larger number of classes, use a simpler plot
        sns.heatmap(cm, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

    plt.show()
    return fig


def main():
    pass


if __name__ == "__main__":
    main()
