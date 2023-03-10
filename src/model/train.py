import os.path
from typing import List, Dict

import hydra
import torch
import torch.nn as nn
import numpy as np

from omegaconf import DictConfig
from matplotlib import pyplot as plt

from src.model.model import PieceClassifier
from src.model.evaluate import eval_model
from src.model.dataset import PiecesDataset
from src.model.consts import TRAIN_CONFIG_PATH, TRAIN_CONFIG_NAME


def train(config: DictConfig) -> (str, torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """A train script for the model over the chess pieces dataset

    Args:
        config: hydra config manager

    Returns:
        model_path: path to the saved model
        train_loader: train loader which was used to train the model
        test_loader: test loader with test data

    """
    train_size_ratio = config.hyperparams.train.train_size
    batch_size = config.hyperparams.train.batch_size
    num_workers = config.hyperparams.train.num_workers
    lr = config.hyperparams.train.lr
    num_epochs = config.hyperparams.train.num_epochs
    print_interval = config.hyperparams.train.print_interval
    random_seed = config.hyperparams.train.random_seed
    minibatch_size = config.hyperparams.train.minibatch_size
    shuffle_data = config.hyperparams.train.shuffle_data
    model_path = config.paths.model_paths.model_path
    current_run_path = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']

    images_dir_path = config.paths.data_paths.image_dir_path
    labels_path = config.paths.data_paths.labels_json_path

    is_minibatch = minibatch_size > 0

    transforms = [hydra.utils.instantiate(transform, _convert_='partial') for transform in config.transforms.values()]

    dataset = PiecesDataset(images_dir_path=images_dir_path,
                            labels_path=labels_path,
                            transforms=transforms,
                            minibatch_size=minibatch_size)

    train_size = int(train_size_ratio * len(dataset))
    test_size = len(dataset) - train_size
    if is_minibatch:
        train_dataset = dataset
        eval_train_loader = get_subset_dataloader(dataset=dataset,
                                                  subset_ratio=config.hyperparams.train.eval_train_size)
        test_loader = None
        eval_val_loader = None
    else:
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                                    generator=torch.Generator().manual_seed(
                                                                        random_seed))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=shuffle_data, num_workers=num_workers)

        eval_train_loader = get_subset_dataloader(dataset=train_dataset,
                                                  subset_ratio=config.hyperparams.train.eval_train_size)
        eval_val_loader = get_subset_dataloader(dataset=test_dataset,
                                                subset_ratio=config.hyperparams.train.eval_val_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=shuffle_data, num_workers=num_workers)

    model = PieceClassifier(in_channels=config.model_params.in_channels,
                            hidden_dim=config.model_params.hidden_dim,
                            out_channels=config.model_params.out_channels,
                            num_type_classes=config.model_params.num_type_classes,
                            num_color_classes=config.model_params.num_color_classes)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    type_criterion = nn.NLLLoss()
    color_criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('Starting training')
    epoch_losses = []
    epoch_train_accuracy = {'type': [], 'color': []}
    epoch_val_accuracy = {'type': [], 'color': []}
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        interval_loss = 0.0
        count = 0
        model.train()
        for iter_num, data in enumerate(train_loader):
            images, type_labels, color_labels, is_piece = data

            optimizer.zero_grad()
            type_pred, color_pred = model(images.to(device))
            images.detach().cpu()

            loss = type_criterion(type_pred, type_labels.argmax(dim=1))

            is_piece_idx = is_piece.nonzero()
            if len(is_piece_idx) > 0:
                loss += color_criterion(color_pred[is_piece_idx], color_labels[is_piece_idx].argmax(dim=1))

            loss.backward()
            optimizer.step()

            interval_loss += loss.item()
            epoch_loss += loss.item()

            if iter_num % print_interval == 0 and iter_num > 0:
                print(f'epoch: {epoch}, iteration: {iter_num}, loss: {interval_loss / print_interval:.3f}')
                interval_loss = 0.0

            count += batch_size

        epoch_train_type_accuracy, epoch_train_color_accuracy = eval_model(model, eval_train_loader, 'train',
                                                                           verbose=False, eval_size=0.01)
        epoch_val_type_accuracy, epoch_val_color_accuracy = eval_model(model, eval_val_loader, 'val',
                                                                       verbose=False, eval_size=0.05)

        epoch_train_accuracy['type'].append(epoch_train_type_accuracy)
        epoch_train_accuracy['color'].append(epoch_train_color_accuracy)
        if not is_minibatch:
            epoch_val_accuracy['type'].append(epoch_val_type_accuracy)
            epoch_val_accuracy['color'].append(epoch_val_color_accuracy)

        epoch_loss = epoch_loss / train_size
        epoch_losses.append(epoch_loss)
        print(f'epoch {epoch} loss: {epoch_loss:.3f}\n')

    if is_minibatch:
        plot_learning_curves(epoch_losses, epoch_train_accuracy)
    else:
        plot_learning_curves(epoch_losses, epoch_train_accuracy, epoch_val_accuracy)

    print('Finished training\n')
    print(f'Saving model to {model_path}')

    run_model_path = os.path.join(current_run_path, model_path)
    model_dir_path = os.path.dirname(run_model_path)
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)

    model_scripted = torch.jit.script(model)
    model_scripted.save(run_model_path)

    return run_model_path, train_loader, test_loader


def get_subset_dataloader(dataset: torch.utils.data.Dataset, subset_ratio: float) -> torch.utils.data.DataLoader:
    subset_size = int(subset_ratio * len(dataset))
    random_indices = np.random.choice(np.arange(len(dataset)), size=subset_size, replace=False)
    subset = torch.utils.data.Subset(dataset, random_indices)
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=subset_size)

    return subset_loader


def plot_learning_curves(epoch_losses: List[float], epoch_train_accuracy: Dict[str, List[float]],
                         epoch_val_accuracy: Dict[str, List[float]] = None):
    num_epochs = len(epoch_losses)

    plt.plot(np.arange(len(num_epochs)), epoch_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('NLLLoss over epochs')
    plt.show()

    plt.plot(np.arange(len(num_epochs)), epoch_train_accuracy['type'], label='train')
    if epoch_val_accuracy:
        plt.plot(np.arange(len(num_epochs)), epoch_val_accuracy['type'], label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Balanced accuracy on piece types over epochs')
    plt.show()

    plt.plot(np.arange(len(num_epochs)), epoch_train_accuracy['color'], label='train')
    if epoch_val_accuracy:
        plt.plot(np.arange(len(num_epochs)), epoch_val_accuracy['color'], label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Balanced accuracy on piece colors over epochs')
    plt.show()


@hydra.main(config_path=TRAIN_CONFIG_PATH, config_name=TRAIN_CONFIG_NAME, version_base='1.2')
def run_train_eval(config: DictConfig):
    model_path_, train_loader_, test_loader_ = train(config=config)

    model_ = torch.jit.load(model_path_)


if __name__ == "__main__":
    run_train_eval()
