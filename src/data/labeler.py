from typing import Tuple

from consts.path_consts import DIRS_TO_PARSE_NAMES


def get_piece_labels(x: int, y: int, board_type: str) -> Tuple[int, int]:
    piece_type = 'X'
    piece_color = 'X'

    if x not in [1, 2, 7, 8]:
        piece_type = 'X'

    elif board_type == str(DIRS_TO_PARSE_NAMES[0]):
        if x == 2:
            piece_type = 'P'
            piece_color = 'W'
        elif x == 7:
            piece_type = 'P'
            piece_color = 'B'
        else:
            if x == 1:
                piece_color = 'W'
            elif x == 8:
                piece_color = 'B'

            if y == 1 or y == 8:
                piece_type = 'R'
            elif y == 2 or y == 7:
                piece_type = 'N'
            elif y == 3 or y == 6:
                piece_type = 'B'
            elif y == 4:
                piece_type = 'Q'
            elif piece_type == 5:
                piece_type = 'K'

    elif board_type == str(DIRS_TO_PARSE_NAMES[1]):

        if x == 1:
            piece_color = 'W'
        elif x == 8:
            piece_color = 'B'

        if y == 3:
            piece_type += 'K'
        elif y == 4:
            piece_type += 'Q'

    return piece_type, piece_color
