import hydra
import torch
import torch.nn as nn

from omegaconf import DictConfig

from src.model.dataset import PiecesDataset
from src.model.consts import TRAIN_CONFIG_PATH, TRAIN_CONFIG_NAME
from src.data.consts.piece_consts import REVERSED_PIECE_TYPE, REVERSED_PIECE_COLOR


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


def eval_model(model, loader: torch.data.utils.DataLoader, state: str):
    """Evaluates the per-class type and color accuracy, as well as a balanced accuracy for type and class

    Args:
        model: trained model
        loader: data loader with a dataset to test the performance of the model over
        state: the type of dataset the model is run on (train or test)

    """
    with torch.no_grad():
        num_piece_classes = len(REVERSED_PIECE_TYPE)
        num_piece_colors = len(REVERSED_PIECE_COLOR) - 1

        type_correct_hits = torch.zeros(num_piece_classes)
        type_counts = torch.zeros(num_piece_classes)
        color_correct_hits = torch.zeros(num_piece_colors)
        color_counts = torch.zeros(num_piece_colors)

        for image, type_label, color_label, is_piece in loader:
            type_pred_probs, color_pred_probs = model(image)
            type_pred = torch.argmax(type_pred_probs, axis=1)
            color_pred = torch.argmax(color_pred_probs, axis=1)

            type_label = torch.argmax(type_label, axis=1)
            color_label = torch.argmax(color_label, axis=1)

            type_correct_hits[type_label] += (type_pred == type_label).to(torch.int64)
            type_counts[type_label] += 1

            is_piece_idx = is_piece.nonzero()

            color_correct_hits[is_piece_idx][color_label] += \
                (color_pred[is_piece_idx] == color_label[is_piece_idx]).to(torch.int64)
            color_counts[is_piece_idx][color_label] += 1

        type_accuracy = type_correct_hits / type_counts
        color_accuracy = color_correct_hits / color_counts

        print(f'\nResults over the {state} set\n')
        for piece_type, piece_name in REVERSED_PIECE_TYPE.items():
            print(f'Accuracy for {piece_name}: {type_accuracy[piece_type]}')

        for piece_color, color_name in REVERSED_PIECE_COLOR.items():
            print(f'Accuracy for {color_name}: {color_accuracy[piece_color]}')

        type_rates = type_counts / type_counts.sum()
        color_rates = color_counts / color_counts.sum()
        balanced_type_accuracy = type_accuracy * type_rates
        balanced_color_accuracy = color_accuracy * color_rates

        print()
        print(f'Balanced type accuracy:', balanced_type_accuracy)
        print(f'Balanced color accuracy:', balanced_color_accuracy)


if __name__ == "__main__":
    model_path_, train_loader_, test_loader_ = train()

    model = torch.jit.load(model_path_)
    model.eval()

    eval_model(model, train_loader_, state='train')
    eval_model(model, test_loader_, state='test')

