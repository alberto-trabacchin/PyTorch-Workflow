import torch
import argparse
from torch.utils.data import DataLoader

import data_setup, model_builder, engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type = str, default = "training_run", help = "Name of the training run")
    parser.add_argument("--n_samples", type = int, default = 1000, help = "Number of samples to generate")
    parser.add_argument("--n_classes", type = int, default = 3, help = "Number of classes to generate")
    parser.add_argument("--n_features", type = int, default = 100, help = "Number of features to generate")
    parser.add_argument("--labeled_size", type = int, default = 100, help = "Number of labeled samples to generate")
    parser.add_argument("--unlabeled_size", type = int, default = 700, help = "Number of unlabeled samples to generate")
    parser.add_argument("--batch_size", type = int, default = 32, help = "Number of samples per batch")
    parser.add_argument("--lr", type = float, default = 0.1, help = "Learning rate for optimizer")
    parser.add_argument("--epochs", type = int, default = 10, help = "Number of epochs to train the model for")
    parser.add_argument("--num_workers", type = int, default = 1, help = "Number of workers for DataLoader")
    parser.add_argument("--device", type = torch.device, default = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), help = "Device to train model on")
    parser.add_argument("--seed", type = int, default = 43, help = "Seed for reproducibility")
    parser.add_argument("--wandb_mode", type = str, choices = ["online", "offline", "disabled"], default = "disabled", help = "Wandb mode")
    parser.add_argument("--save_chkpt", type = int, default = None, help = "Number of epochs between checkpoints")
    parser.add_argument("--load_chkpt", type = int, default = None, help = "Epoch to load checkpoint from")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Create labeled, unlabeled, and test datasets
    lab_dataset, unlab_dataset, test_dataset = data_setup.get_datasets(
        n_samples = args.n_samples,
        n_classes = args.n_classes,
        n_features = args.n_features,
        lab_size = args.labeled_size,
        unlab_size = args.unlabeled_size,
        random_state = args.seed,
        shuffle = True
    )

    # Create dataloaders
    lab_loader = DataLoader(
        dataset = lab_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers
    )
    unlab_loader = DataLoader(
        dataset = unlab_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers
    )
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers
    )

    # Create model
    model = model_builder.LinearModel(
        input_dim = args.n_features,
        output_dim = args.n_classes
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
    if args.load_chkpt is not None:
        model, optimizer = engine.load_checkpoint(
            model_name = args.model_name,
            model = model,
            optimizer = optimizer,
            epoch = args.load_chkpt
        )

    # Train model
    results = engine.train(
        model = model,
        model_name = args.model_name,
        train_dataloader = lab_loader,
        test_dataloader = test_loader,
        loss_fn = loss_fn,
        optimizer = optimizer,
        epochs = args.epochs,
        device = args.device,
        save_chkpt = args.save_chkpt,
        load_chkpt = args.load_chkpt
    )
