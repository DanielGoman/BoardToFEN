import logging
import torch

from torch.utils.tensorboard import SummaryWriter

from src.data.consts.piece_consts import REVERSED_LABELS, LABELS
from src.visualization.tensorboard_visualization_utils import plot_classes_preds


def eval_model(model, criterion, loader: torch.utils.data.DataLoader, device: str, state: str = 'Full eval',
               log: logging.Logger = None, epoch_num: int = None, tb_writer: SummaryWriter = None):
    """Evaluates the per-class type and color accuracy, as well as a balanced accuracy for type and class

    Args:
        model: trained model
        loader: data loader with a dataset to test the performance of the model over
        state: the type of dataset the model is run on (train or test)
        log: logger object, print and saves log into the logger if an object is passed, otherwise silent
        epoch_num: number of the current epoch
        tb_writer: TensorBoard writer object, used to log results per epoch

    Returns:
        balanced_class_accuracy: Balanced accuracy score on the given loader
        loss: loss on the given loader

    """
    model.eval()
    with torch.no_grad():
        loss = 0
        num_classes = len(LABELS)
        confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        for i, (images, labels) in enumerate(loader):
            output_scores = model(images.to(device)).cpu()
            loss_i = criterion(output_scores, labels)
            class_preds = torch.argmax(torch.softmax(output_scores, dim=1), dim=1).cpu()

            tb_writer.add_figure(f'Evaluation on {state}, epoch {epoch_num}, eval iteration {i}',
                                 plot_classes_preds(images, output_scores.cpu(), labels))
            images.cpu()

            class_labels = torch.argmax(labels, dim=1)

            loss += loss_i.item()
            for pred_i, label_i in zip(class_preds, class_labels):
                confusion_matrix[label_i, pred_i] += 1

        loss /= len(loader)
        class_counts = torch.sum(confusion_matrix, dim=1)
        class_accuracy = torch.nan_to_num(torch.diag(confusion_matrix) / class_counts)
        class_accuracy[class_accuracy == float("inf")] = 0

        class_rates = class_counts / class_counts.sum()
        balanced_class_accuracy = class_accuracy @ class_rates

        if log:
            log.info(f'\nResults over the {state} set\n')
            for piece_class, piece_name in REVERSED_LABELS.items():
                log.info(f'Accuracy for {piece_name}: {class_accuracy[piece_class]:.3f}')

            log.info('')
            log.info(f'Balanced class accuracy on {state}: {balanced_class_accuracy.item():.3f}')

        return balanced_class_accuracy.item(), loss
