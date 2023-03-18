import logging

import torch
from torch.utils.tensorboard import SummaryWriter

from src.data.consts.piece_consts import REVERSED_PIECE_TYPE, REVERSED_PIECE_COLOR


def eval_model(model, loader: torch.utils.data.DataLoader, state: str, log: logging.Logger = None,
               epoch_num: int = None, tb_writer: SummaryWriter = None):
    """Evaluates the per-class type and color accuracy, as well as a balanced accuracy for type and class

    Args:
        model: trained model
        loader: data loader with a dataset to test the performance of the model over
        state: the type of dataset the model is run on (train or test)
        log: logger object, print and saves log into the logger if an object is passed, otherwise silent
        epoch_num: number of the current epoch
        tb_writer: TensorBoard writer object, used to log results per epoch

    """
    model.eval()
    with torch.no_grad():
        num_piece_classes = len(REVERSED_PIECE_TYPE)
        num_piece_colors = len(REVERSED_PIECE_COLOR)

        type_correct_hits = torch.zeros(num_piece_classes)
        type_counts = torch.zeros(num_piece_classes)
        color_correct_hits = torch.zeros(num_piece_colors)
        color_counts = torch.zeros(num_piece_colors)

        for i, (image, type_label, color_label, is_piece) in enumerate(loader):
            type_pred_probs, color_pred_probs = model(image)
            type_pred = torch.argmax(type_pred_probs, axis=1)
            color_pred = torch.argmax(color_pred_probs, axis=1)

            type_label = torch.argmax(type_label, axis=1)
            color_label = torch.argmax(color_label, axis=1)

            type_correct_hits[type_label] += (type_pred == type_label).to(torch.int64)
            labels_count_per_class = torch.bincount(type_label)
            type_counts[labels_count_per_class.nonzero()] += labels_count_per_class[labels_count_per_class.nonzero()]

            is_piece_idx = is_piece.nonzero()

            if len(is_piece_idx) > 0:
                color_correct_hits[color_label[is_piece_idx]] += \
                    (color_pred[is_piece_idx] == color_label[is_piece_idx]).to(torch.int64)
                color_count_per_class = torch.bincount(color_label)
                color_counts[color_count_per_class.nonzero()] += color_count_per_class[color_count_per_class.nonzero()]

        type_accuracy = torch.nan_to_num(type_correct_hits / type_counts)
        color_accuracy = torch.nan_to_num(color_correct_hits / color_counts)

        type_rates = type_counts / type_counts.sum()
        color_rates = color_counts / color_counts.sum()
        balanced_type_accuracy = type_accuracy @ type_rates
        balanced_color_accuracy = color_accuracy @ color_rates

        if log:
            log.info(f'\nResults over the {state} set\n')
            for piece_type, piece_name in REVERSED_PIECE_TYPE.items():
                log.info(f'Accuracy for {piece_name}: {type_accuracy[piece_type]:.3f}')

            for piece_color, color_name in REVERSED_PIECE_COLOR.items():
                log.info(f'Accuracy for {color_name}: {color_accuracy[piece_color]:.3f}')

            log.info('')
            log.info(f'Balanced type accuracy: {balanced_type_accuracy.item():.3f}')
            log.info(f'Balanced color accuracy: {balanced_color_accuracy.item():.3f}')

        if tb_writer:
            tb_writer.add_scalar('Balanced type accuracy on train', balanced_type_accuracy.item(), epoch_num)
            tb_writer.add_scalar('Balanced color accuracy on train', balanced_color_accuracy.item(), epoch_num)
            tb_writer.flush()

        return balanced_type_accuracy, balanced_color_accuracy
