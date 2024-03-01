from itertools import groupby
from typing import List, Tuple

import numpy as np

from src.data.consts.piece_consts import LABELS, REVERSED_LABELS
from src.data.consts.squares_consts import BOARD_SIDE_SIZE


def convert_row_to_fen(row: np.ndarray) -> str:
    """Converts a board row to a fen representation of that row

    Args:
        row: row of integers representing pieces in the model's predictions format

    Returns:
        fen: a fen representation of a row of a chess board

    """
    start_indices, sequence_lengths = find_sequence_indices(row, value_to_count=LABELS['X'])

    idx = 0
    fen_row = []
    while idx < BOARD_SIDE_SIZE:
        if idx in start_indices:
            sequence_idx = start_indices.index(idx)
            sequence_length = sequence_lengths[sequence_idx]
            fen_row.append(str(sequence_length))
            idx += sequence_length
        else:
            piece_label = REVERSED_LABELS[row[idx]]
            fen_row.append(piece_label)
            idx += 1

    fen = ''.join(fen_row)
    return fen


def find_sequence_indices(arr: np.ndarray, value_to_count: int) -> Tuple[List[int], List[int]]:
    """Finds the start index and length of every sequence of the `value_to_count` value

    Args:
        arr: 1d array
        value_to_count: the value of which we're interested in finding its sequences in `arr`

    Returns:
        start_indices: list of the first index of each sequence
        sequence_lengths: list of lengths of each sequence found

    """
    start_indices = []
    sequence_lengths = []

    # group-by matching sequences and non-matching sequences
    for is_desired_sequence, sequence_iterator in groupby(enumerate(arr == value_to_count), key=lambda x: x[1]):
        # skips non-matching sequences
        if is_desired_sequence:
            # (enumerate-index, value inside the sequence)
            start_index, _ = next(sequence_iterator)
            # include the starting element we just dropped using `next`
            length = len(list(sequence_iterator)) + 1

            start_indices.append(start_index)
            sequence_lengths.append(length)

    return start_indices, sequence_lengths
