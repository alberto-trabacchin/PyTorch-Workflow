import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from pathlib import Path

import utils


def load_checkpoint(
        model_name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int
) -> Dict:
    """Loads a PyTorch model checkpoint.

    Loads a PyTorch model checkpoint from a specified path.

    Args:
        model_name: A string indicating the name of the model.
        epoch: An integer indicating the epoch of the checkpoint.

    Returns:
        A PyTorch model state_dict.
    """
    model_path = Path(f"models/{model_name}/{model_name}_epoch{epoch}.pth")
    checkpt = torch.load(model_path)
    model.load_state_dict(checkpt["model_state_dict"])
    optimizer.load_state_dict(checkpt["optimizer_state_dict"])
    epoch = checkpt["epoch"]
    return model, optimizer
    

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        save_path: Path
) -> None:
    """Saves a PyTorch model checkpoint.

    Saves a PyTorch model checkpoint to a specified path.

    Args:
        model: A PyTorch model to be saved.
        optimizer: A PyTorch optimizer to be saved.
        epoch: An integer indicating the current epoch.
        path: A Path object indicating where to save the checkpoint.
    """
    save_path.parent.mkdir(parents = True, exist_ok = True)
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, save_path)


def train_step(
        model: torch.nn.Module,
        data_loader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """
    model.train()
    train_loss = 0
    train_accuracy = 0
    for batch_sample in data_loader:
        X, y = batch_sample
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_accuracy += utils.accuracy(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)
    return train_loss, train_accuracy


def test_step(
        model: torch.nn.Module,
        data_loader: DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device,
) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
    """
    model.eval()
    test_loss = 0
    test_accuracy = 0
    with torch.inference_mode():
        for batch_sample in data_loader:
            X, y = batch_sample
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            test_accuracy += utils.accuracy(y_pred, y)
        test_loss /= len(data_loader)
        test_accuracy /= len(data_loader)
    return test_loss, test_accuracy


def train(
        model: torch.nn.Module,
        model_name: str,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: torch.device,
        save_chkpt: int = None,
        load_chkpt: int = None
) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]} 
        For example if training for epochs=2: 
                    {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]} 
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epoch": []
    }
    train_accuracy = 0
    test_accuracy = 0
    train_loss = 0
    test_loss = 0
    if load_chkpt is None:
        load_chkpt = 0
    for epoch in tqdm(range(epochs), initial = load_chkpt):
        model = model.to(device)
        train_loss, train_accuracy = train_step(
            model = model,
            data_loader = train_dataloader,
            loss_fn = loss_fn,
            optimizer = optimizer,
            device = device
        )
        test_loss, test_accuracy = test_step(
            model = model,
            data_loader = test_dataloader,
            loss_fn = loss_fn,
            device = device
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_accuracy)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_accuracy)
        results["epoch"].append(epoch + 1)

        if save_chkpt is not None:
            if ((epoch + 1) % save_chkpt) == 0:
                save_checkpoint(
                    model = model,
                    optimizer = optimizer,
                    epoch = epoch + 1,
                    save_path = Path(f"models/{model_name}/{model_name}_epoch{epoch + 1}.pth")
                )
    return results