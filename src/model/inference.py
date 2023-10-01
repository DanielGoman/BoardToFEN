import torch
import numpy as np

from typing import List, Tuple

from src.model.dataset import PiecesDataset
from src.model.model import PieceClassifier


# TODO: consider adding separate config for inference mode
def inference(model: PieceClassifier, board_squares: List[List[np.ndarray]]) -> np.ndarray:
    """Performs inference on a list of a selected board's squares using a trained model

    Args:
        model: trained model for square classification
        board_squares: 2d list of squares of the board

    Returns:
        2d array of predicted numerical labels as in src/data/consts/piece_consts.py

    """
    pieces_dataset, num_squares = board_squares_to_pieces_dataset(board_squares)
    pieces_loader = torch.utils.data.DataLoader(pieces_dataset,
                                                batch_size=num_squares,
                                                shuffle=False,
                                                num_workers=4)

    predicted_labels = model.inference(pieces_loader.next())
    return predicted_labels


def board_squares_to_pieces_dataset(board_squares: List[List[np.ndarray]]) -> Tuple[PiecesDataset, int]:
    """Transforms the 2d list of squares into PiecesDataset

    Args:
        board_squares: 2d list of squares of the board

    Returns:
        - PiecesDataset that contains all the squares from `board_squares`
        - number of squares on the board

    """
    pass

