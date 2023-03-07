import yaml

import torch
import torch.nn as nn

from src.model.dataset import PiecesDataset
from src.model.consts import CONFIG_PATH
from src.data.consts.piece_consts import REVERSED_PIECE_TYPE


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

    model = torch.jit.load(model_path)
    model.eval()





def eval_model(model, loader: torch.data.utils.DataLoader, state: str):
    with torch.no_grad():
        num_piece_classes = len(REVERSED_PIECE_TYPE)

        type_accuracy = torch.zeros(num_piece_classes)
        type_counts = torch.zeros(num_piece_classes)
        color_accuracy = torch.zeros(num_piece_classes)
        color_counts = torch.zeros(num_piece_classes)

        for image, type_label, color_label, is_piece in loader:
            type_pred_probs, color_pred_probs = model(image)
            type_pred = torch.argmax(type_pred_probs, axis=1)
            color_pred = torch.argmax(color_pred_probs, axis=1)

            type_label = torch.argmax(type_label, axis=1)
            color_label = torch.argmax(color_label, axis=1)

            type_accuracy[type_label] += (type_pred == type_label).to(torch.int64)
            type_counts[type_label] += 1

            # TODO: evaluate accuracy for color. Issue is that sometimes there is no piece on the board (empty square)
            color_counts = pass





if __name__ == "__main__":
    train()
