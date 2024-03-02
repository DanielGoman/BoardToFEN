import os
from typing import List

import numpy as np
from matplotlib import pyplot as plt


def plot_learning_curves(epoch_losses: List[float], epoch_train_accuracies: List[float],
                         epoch_val_accuracies: List[float] = None, plots_path: str = ''):
    """Plots train loss, train accuracy and validation accuracy over epochs

    Args:
        epoch_losses: average accumulated loss per epoch
        epoch_train_accuracies: train accuracy per epoch (type and color separately)
        epoch_val_accuracies: validation accuracy per epoch (type and color separately)
        plots_path: path to the directory in which the plots will be saved

    """
    num_epochs = len(epoch_losses)

    plt.plot(np.arange(num_epochs), epoch_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('NLLLoss over epochs')
    plt.savefig(os.path.join(plots_path, 'loss.png'))
    plt.show()

    plt.plot(np.arange(num_epochs), epoch_train_accuracies, label='train')
    if epoch_val_accuracies:
        plt.plot(np.arange(num_epochs), epoch_val_accuracies, label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Balanced accuracy on piece types over epochs')
    plt.legend()
    plt.savefig(os.path.join(plots_path, 'train_accuracy.png'))
    plt.show()