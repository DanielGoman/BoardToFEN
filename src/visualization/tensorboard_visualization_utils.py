import numpy as np
import torch

from matplotlib import pyplot as plt

from src.data.consts.piece_consts import REVERSED_LABELS
from src.visualization.consts import NUM_DISPLAYED_IMAGES_PER_ROW


def plot_classes_preds(images: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor):
    """Generates a grid of input images, their respective GT labels and predicted labels

    Args:
        images: tensor of input images
        preds: predicted score on the input images
        labels: GT labels of the input images (as onehot vectors)

    Returns:
        fig: image frid with the prediction scores and respective GT labels

    """
    predicted_labels, pred_probs = images_to_probs(preds)
    num_preds = len(preds)
    num_rows = num_preds // NUM_DISPLAYED_IMAGES_PER_ROW + 1 if num_preds % NUM_DISPLAYED_IMAGES_PER_ROW \
        else num_preds // NUM_DISPLAYED_IMAGES_PER_ROW

    # plot the images in the batch, along with predicted and true labels
    fig, axs = plt.subplots(num_rows, NUM_DISPLAYED_IMAGES_PER_ROW, figsize=(8, 4))
    for idx in np.arange(num_preds):
        ax = axs[idx // NUM_DISPLAYED_IMAGES_PER_ROW, idx % NUM_DISPLAYED_IMAGES_PER_ROW]
        ax.set_xticks([])
        ax.set_yticks([])
        label = torch.nonzero(labels[idx])[0, 0].item()
        matplotlib_imshow(ax, images[idx], one_channel=False)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            REVERSED_LABELS[predicted_labels[idx]],
            pred_probs[idx] * 100.0,
            REVERSED_LABELS[label]),
            color=("green" if predicted_labels[idx] == label else "red"),
        )

    # hide unnecessary extra axes
    if num_rows * NUM_DISPLAYED_IMAGES_PER_ROW > num_preds:
        for idx in range(num_preds % NUM_DISPLAYED_IMAGES_PER_ROW, NUM_DISPLAYED_IMAGES_PER_ROW):
            axs[num_rows - 1, idx].set_visible(False)

    plt.tight_layout()
    return fig


def images_to_probs(preds: torch.Tensor):
    """Converts the predicted scores to predicted labels using softmax

    Args:
        preds: predicted scores over a set of images

    Returns:
        predicted_labels: the predicted labels for each input image
        predicted_probs: the predicted probability for the predicted label

    """
    # convert output probabilities to predicted class
    predicted_labels = torch.argmax(preds, 1)
    predicted_labels = np.squeeze(predicted_labels.numpy())
    predicted_probs = [torch.softmax(pred_output, dim=0)[predicted_label].item()
                       for predicted_label, pred_output in zip(predicted_labels, preds)]
    return predicted_labels, predicted_probs


def matplotlib_imshow(ax: plt.Axes, img: torch.Tensor, one_channel: bool = False):
    """Displays the input image in greyscale mode or RGB image

    Args:
        ax: the axis to display the image on
        img: the image to display
        one_channel: displays the images in greyscale mode if True, else displays it as an RGB image

    """
    if one_channel:
        img = img.mean(dim=0)
    np_img = img.numpy()
    if one_channel:
        ax.imshow(np_img, cmap="Greys")
    else:
        ax.imshow(np.transpose(np_img, (1, 2, 0)))
