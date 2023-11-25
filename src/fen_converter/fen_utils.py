from itertools import groupby
from typing import List, Tuple

import numpy as np

from src.data.consts.piece_consts import LABELS, REVERSED_LABELS
from src.data.consts.squares_consts import BOARD_SIDE_SIZE


def convert_row_to_fen(row: np.ndarray) -> str:
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
    start_indices = []
    sequence_lengths = []

    for k, g in groupby(enumerate(arr == value_to_count), key=lambda x: x[1]):
        if k:
            start_index, _ = next(g)
            length = len(list(g)) + 1  # Include the starting element
            start_indices.append(start_index)
            sequence_lengths.append(length)

    return start_indices, sequence_lengths


if __name__ == "__main__":
    arr = np.array([11, 12, 0, 12, 12, 0, 12, 0])

    print(convert_row_to_fen(arr))


