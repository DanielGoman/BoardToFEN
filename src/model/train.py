import os.path

import hydra
import torch
import torch.nn as nn

from omegaconf import DictConfig

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
    model_path = config.paths.model_paths.model_path
    current_run_path = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']

    images_dir_path = config.paths.data_paths.image_dir_path
    labels_path = config.paths.data_paths.labels_json_path

    transforms = [hydra.utils.instantiate(transform, _convert_='partial') for transform in config.transforms.values()]

    dataset = PiecesDataset(images_dir_path=images_dir_path,
                            labels_path=labels_path,
                            transforms=transforms,
                            minibatch_size=minibatch_size)

    train_size = int(train_size_ratio * len(dataset))
    test_size = len(dataset) - train_size
    if minibatch_size > 0:
        train_dataset = dataset
        test_loader = None
    else:
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                                    generator=torch.Generator().manual_seed(random_seed))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

    model = PieceClassifier(in_channels=config.model_params.in_channels,
                            hidden_dim=config.model_params.hidden_dim,
                            out_channels=config.model_params.out_channels,
                            num_type_classes=config.model_params.num_type_classes,
                            num_color_classes=config.model_params.num_color_classes)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('Starting training')
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        interval_loss = 0.0
        for iter_num, data in enumerate(train_loader):
            image, type_label, color_label, is_piece = data

            optimizer.zero_grad()

            type_pred, color_pred = model(image.to(device))
            image.detach().cpu()

            loss = criterion(type_pred, type_label.to(device))

            is_piece_idx = is_piece.nonzero()
            if len(is_piece_idx) > 0:
                loss += criterion(color_pred[is_piece_idx], color_label[is_piece_idx])

            loss.backward()
            optimizer.step()

            interval_loss += loss.item()
            epoch_loss += loss.item()

            if iter_num % print_interval == 0:
                print(f'epoch: {epoch}, iteration: {iter_num}, loss: {interval_loss / print_interval:.3f}')
                interval_loss = 0.0

        print(f'epoch {epoch} loss: {epoch_loss / train_size:.3f}\n')

    print('Finished training\n')
    print(f'Saving model to {model_path}')

    run_model_path = os.path.join(current_run_path, model_path)
    model_dir_path = os.path.dirname(run_model_path)
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)

    model_scripted = torch.jit.script(model)
    model_scripted.save(model_dir_path)

    return model_dir_path, train_loader, test_loader


@hydra.main(config_path=TRAIN_CONFIG_PATH, config_name=TRAIN_CONFIG_NAME, version_base='1.2')
def run_train_eval(config: DictConfig):
    model_path_, train_loader_, test_loader_ = train(config=config)

    model_ = torch.jit.load(model_path_)
    model_.eval()

    eval_model(model_, train_loader_, state='train')
    if config.hyperparams.train.minibatch_size == -1:
        eval_model(model_, test_loader_, state='test')


if __name__ == "__main__":
    run_train_eval()

