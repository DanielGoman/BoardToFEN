import yaml

import torch
import torch.nn as nn

from src.model.dataset import PiecesDataset
from src.model.consts import CONFIG_PATH


def train():
    yaml_file = open(CONFIG_PATH)
    configs = yaml.safe_load(yaml_file)
    data_configs = configs['DATA']
    train_configs = configs['TRAIN']

    train_size_ratio = train_configs['train_size']
    batch_size = train_configs['batch_size']
    num_workers = train_configs['num_workers']
    lr = train_configs['lr']
    momentum = train_configs['momentum']
    num_epochs = train_configs['num_epochs']
    print_interval = train_configs['print_interval']
    model_path = train_configs['model_path']

    dataset = PiecesDataset(images_dir_path=data_configs['IMAGE_DIR_PATH'],
                            labels_path=data_configs['LABELS_JSON_PATH'])

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
            if is_piece:
                loss += criterion(color_pred, color_label)

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
