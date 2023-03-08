import hydra
import torch
import torch.nn as nn

from omegaconf import DictConfig

from src.model.evaluate import eval_model
from src.model.dataset import PiecesDataset
from src.model.consts import TRAIN_CONFIG_PATH, TRAIN_CONFIG_NAME


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
    train_size_ratio = config.hyperparams.train.train_size
    batch_size = config.hyperparams.train.batch_size
    num_workers = config.hyperparams.train.num_workers
    lr = config.hyperparams.train.lr
    momentum = config.hyperparams.train.momentum
    num_epochs = config.hyperparams.train.num_epochs
    print_interval = config.hyperparams.train.print_interval
    model_path = config.paths.model_paths.model_path

    images_dir_path = config.paths.data_paths.image_dir_path
    labels_path = config.paths.data_paths.labels_json_path

    transforms = [hydra.utils.instantiate(transform) for transform in config.transforms.values()]

    dataset = PiecesDataset(images_dir_path=images_dir_path,
                            labels_path=labels_path,
                            transforms=transforms)

    train_size = int(train_size_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)

    # TODO: replace this with initialization of the model
    model = None

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, momentum=momentum)

    print('Starting training')
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        interval_loss = 0.0
        for iter_num, data in enumerate(train_loader):
            image, type_label, color_label, is_piece = data

            optimizer.zero_grad()

            type_pred, color_pred = model(image)

            loss = criterion(type_pred, type_label)

            is_piece_idx = is_piece.nonzero()
            loss += criterion(color_pred[is_piece_idx], color_label[is_piece_idx])

            loss.backward()
            optimizer.step()

            interval_loss += loss.item()
            epoch_loss += loss.item()

            if iter_num % print_interval == 0:
                print(f'epoch: {epoch}, iteration: {iter_num}, loss: {interval_loss / print_interval}:.3f')
                interval_loss = 0.0

        print(f'epoch {epoch} loss: {epoch_loss / len(train_size)}:.3f\n')

    print('Finished training\n')
    print(f'Saving model to {model_path}')
    model_scripted = torch.jit.script(model)
    model_scripted.save(model_path)

    return model_path, train_loader, test_loader


if __name__ == "__main__":
    model_path_, train_loader_, test_loader_ = train()

    model = torch.jit.load(model_path_)
    model.eval()

    eval_model(model, train_loader_, state='train')
    eval_model(model, test_loader_, state='test')

