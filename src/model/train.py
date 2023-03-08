import hydra
import torch
import torch.nn as nn

from omegaconf import DictConfig

from src.model.dataset import PiecesDataset
from src.model.consts import TRAIN_CONFIG_PATH, TRAIN_CONFIG_NAME


@hydra.main(config_path=TRAIN_CONFIG_PATH, config_name=TRAIN_CONFIG_NAME, version_base='1.2')
def train(config: DictConfig):
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
    print('Saving model')
    model_scripted = torch.jit.script(model)
    model_scripted.save(model_path)


if __name__ == "__main__":
    train()
