import logging

import torch
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter

from src.data.consts.piece_consts import REVERSED_LABELS, LABELS
from src.visualization.tensorboard_visualization_utils import plot_classes_preds


def eval_model(model, loader: torch.utils.data.DataLoader, device: str, state: str = 'Full eval',
               log: logging.Logger = None, epoch_num: int = None, tb_writer: SummaryWriter = None):
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
        num_classes = len(LABELS)

        class_correct_hits = torch.zeros(num_classes)
        class_counts = torch.zeros(num_classes)

        for i, (images, labels) in enumerate(loader):
            # img_grid = torchvision.utils.make_grid(images)
            output = model(images.to(device))
            class_pred = torch.argmax(torch.exp(output), axis=1).cpu()

            tb_writer.add_figure(f'Evaluation on {state}, epoch {epoch_num}, eval iteration {i}',
                                 plot_classes_preds(images, output.cpu(), labels)) #,
                                 # global_step=epoch_num * len(loader) + i)
            images.cpu()

            class_label = torch.argmax(labels, axis=1)

            class_correct_hits[class_label] += (class_pred == class_label).to(torch.int64)
            labels_count_per_class = torch.bincount(class_label)
            class_counts[labels_count_per_class.nonzero()] += labels_count_per_class[labels_count_per_class.nonzero()]

        class_accuracy = torch.nan_to_num(class_correct_hits / class_counts)

        class_rates = class_counts / class_counts.sum()
        balanced_class_accuracy = class_accuracy @ class_rates

        if log:
            log.info(f'\nResults over the {state} set\n')
            for piece_class, piece_name in REVERSED_LABELS.items():
                log.info(f'Accuracy for {piece_name}: {class_accuracy[piece_class]:.3f}')

            log.info('')
            log.info(f'Balanced class accuracy: {balanced_class_accuracy.item():.3f}')

        if tb_writer:
            tb_writer.add_scalar('A - Balanced class accuracy on train', balanced_class_accuracy.item(), epoch_num)
            tb_writer.flush()

        return balanced_class_accuracy
