import torch
import numpy as np

from typing import Tuple, Dict

import torchvision
from PIL import Image

from src.model.model import PieceClassifier


# TODO: consider adding separate config for inference mode
def inference(model: PieceClassifier, board_squares: Dict[Tuple[int, int], np.ndarray],
              transforms: list) -> np.ndarray:
    """Performs inference on a list of a selected board's squares using a trained model

    Args:
        model: trained model for square classification
        board_squares: "2d" dict of squares.
                        board_squares[(square_x, square_y)] = np.ndarray of the square
        transforms: list of transforms to be applied to the squares

    Returns:
        2d array of predicted numerical labels as in src/data/consts/piece_consts.py

    """
    pieces_batch = board_squares_to_pieces_dataset(board_squares, transforms)
    predicted_labels = model.inference(pieces_batch)

    return predicted_labels



