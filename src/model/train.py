import os
import logging
import hydra

import torch
import torchvision
import numpy as np
import torch.nn as nn

from typing import List
from omegaconf import DictConfig
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.model.model import PieceClassifier
from src.model.evaluate import eval_model
from src.model.dataset import PiecesDataset
from src.consts import TRAIN_CONFIG_PATH, TRAIN_CONFIG_NAME
from src.utils.transforms import parse_config_transforms


@hydra.main(config_path=TRAIN_CONFIG_PATH, config_name=TRAIN_CONFIG_NAME, version_base='1.2')
def train(config: DictConfig) -> (str, torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """A train script for the model over the chess pieces dataset

    Args:
        config: hydra config manager

    Returns:
        model_path: path to the saved model
        train_loader: train loader which was used to train the model
        test_loader: test loader with test data

    """
    batch_size = config.hyperparams.train.batch_size
    num_workers = config.hyperparams.train.num_workers
    lr = config.hyperparams.train.lr
    num_epochs = config.hyperparams.train.num_epochs
    print_interval = config.hyperparams.train.print_interval
    minibatch_size = config.hyperparams.train.minibatch_size
    shuffle_data = config.hyperparams.train.shuffle_data
    model_path = config.paths.model_paths.model_path
    plots_dirname = config.paths.plot_paths.plot_dirname

    current_run_path = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    plots_path = os.path.join(current_run_path, plots_dirname)
    os.makedirs(plots_path, exist_ok=True)
    log = logging.getLogger(__name__)
    tb_writer = SummaryWriter()

    images_dir_path = config.paths.data_paths.image_dir_path
    train_labels_path = config.paths.data_paths.train_json_path
    vel_labels_path = config.paths.data_paths.val_json_path

    is_minibatch = minibatch_size > 0

    transforms = parse_config_transforms(config.transforms)

    train_dataset = PiecesDataset(images_dir_path=images_dir_path,
                                  labels_path=train_labels_path,
                                  transforms=transforms)
    val_dataset = PiecesDataset(images_dir_path=images_dir_path,
                                labels_path=vel_labels_path,
                                transforms=transforms)

    if is_minibatch:
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        eval_train_loader = get_subset_dataloader(dataset=train_dataset,
                                                  subset_ratio=config.hyperparams.train.eval_train_size)
        val_loader = None
        eval_val_loader = None
    else:
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                 shuffle=shuffle_data, num_workers=num_workers)

        eval_train_loader = get_subset_dataloader(dataset=train_dataset,
                                                  subset_ratio=config.hyperparams.train.eval_train_size)
        # eval_val_loader = get_subset_dataloader(dataset=val_dataset,
        #                                         subset_ratio=config.hyperparams.train.eval_val_size)
        eval_val_loader = val_loader

    train_size = len(train_dataset) * batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=shuffle_data, num_workers=num_workers)

    model = PieceClassifier(in_channels=config.model_params.in_channels,
                            hidden_dim=config.model_params.hidden_dim,
                            out_channels=config.model_params.out_channels,
                            num_classes=config.model_params.num_classes)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log.info('Starting training')
    epoch_losses = []
    epoch_train_accuracies = []
    epoch_val_accuracies = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        interval_loss = 0.0
        model.train()
        for iter_num, data in enumerate(train_loader):
            images, label = data
            images, label = images.to(device), label.to(device)

            output_scores = model(images)

            loss = criterion(output_scores, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            label.detach().cpu()
            images.detach().cpu()

            interval_loss += loss.item()
            epoch_loss += loss.item()

            if iter_num % print_interval == 0:
                img_grid = torchvision.utils.make_grid(images)
                tb_writer.add_image(f'Epoch {epoch} batch {iter_num}', img_grid)
                if iter_num > 0:
                    log.info(f'epoch: {epoch}, iteration: {iter_num}, loss: '
                             f'{interval_loss / (print_interval * batch_size):.3f}')
                    interval_loss = 0.0

            if 0 < minibatch_size == iter_num:
                break

        epoch_train_accuracy, epoch_train_loss = eval_model(model, criterion, eval_train_loader, device=device,
                                                            state='train', epoch_num=epoch, tb_writer=tb_writer)
        epoch_val_accuracy, epoch_val_loss = eval_model(model, criterion, eval_val_loader, device=device, state='val',
                                                        epoch_num=epoch, tb_writer=tb_writer)
        tb_writer.add_scalars('Balanced class accuracy',
                              {'train': epoch_train_accuracy,
                               'val': epoch_val_accuracy},
                              global_step=epoch)

        tb_writer.add_scalars('Loss',
                              {'train': epoch_train_loss,
                               'val': epoch_val_loss},
                              global_step=epoch)

        epoch_train_accuracies.append(epoch_train_accuracy)
        if not is_minibatch:
            epoch_val_accuracies.append(epoch_val_accuracy)

        epoch_loss = epoch_loss / train_size
        epoch_losses.append(epoch_loss)

        log.info(f'epoch {epoch} loss: {epoch_loss:.3f}\n')

        tb_writer.flush()

    if is_minibatch:
        plot_learning_curves(epoch_losses, epoch_train_accuracies, plots_path=plots_path)
    else:
        plot_learning_curves(epoch_losses, epoch_train_accuracies, epoch_val_accuracies, plots_path=plots_path)

    log.info('Finished training\n')

    if not is_minibatch:
        eval_model(model=model, criterion=criterion, loader=val_loader,
                   device=device, state='val', log=log, tb_writer=tb_writer)

    model_dir_path = os.path.join(current_run_path, os.path.dirname(model_path))
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)

    torch_model_path = os.path.join(current_run_path, model_path)
    log.info(f'Saving model to {torch_model_path}')
    torch.save(model.state_dict(), torch_model_path)

    tb_writer.close()


def get_subset_dataloader(dataset: torch.utils.data.Dataset, subset_ratio: float) -> torch.utils.data.DataLoader:
    """Creates a DataLoader based on a subset of the dataset

    Args:
        dataset: the full dataset
        subset_ratio: the percentage of data that should be sampled into the subset (without repetitions)

    Returns:
        subset_loader: DataLoader over the subset of the given dataset

    """
    subset_size = int(subset_ratio * len(dataset))
    random_indices = np.random.choice(np.arange(len(dataset)), size=subset_size, replace=False)
    subset = torch.utils.data.Subset(dataset, random_indices)
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=subset_size)

    return subset_loader


def plot_learning_curves(epoch_losses: List[float], epoch_train_accuracies: List[float],
                         epoch_val_accuracies: List[float] = None, plots_path: str = ''):
    """Plots train loss, train accuracy and validation accuracy over epochs

    Args:
        epoch_losses: average accumulated loss per epoch
        epoch_train_accuracies: train accuracy per epoch (type and color separately)
        epoch_val_accuracies: validation accuracy per epoch (type and color separately)
        plots_path: path to the directory in which the plots will be saved

    """
    num_epochs = len(epoch_losses)

    plt.plot(np.arange(num_epochs), epoch_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('NLLLoss over epochs')
    plt.savefig(os.path.join(plots_path, 'loss.png'))
    plt.show()

    plt.plot(np.arange(num_epochs), epoch_train_accuracies, label='train')
    if epoch_val_accuracies:
        plt.plot(np.arange(num_epochs), epoch_val_accuracies, label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Balanced accuracy on piece types over epochs')
    plt.legend()
    plt.savefig(os.path.join(plots_path, 'train_accuracy.png'))
    plt.show()


# tensorboard --logdir=src/model/runs
if __name__ == "__main__":
    train()
