

from consts.path_consts import DIRS_TO_PARSE_NAMES


def get_piece_labels(x: int, y: int, board_type: str) -> str:
    label = None

    if x not in [1, 2, 7, 8]:
        return 'X'

    elif board_type == str(DIRS_TO_PARSE_NAMES[0]):
        if x == 2:
            return 'WP'
        elif x == 7:
            return 'BP'
        else:
            if x == 1:
                label = 'W'
            elif x == 8:
                label = 'B'

            if y == 1 or y == 8:
                label += 'R'
            elif y == 2 or y == 7:
                label += 'N'
            elif y == 3 or y == 6:
                label += 'B'
            elif y == 4:
                label += 'Q'
            elif y == 5:
                label += 'K'

    elif board_type == str(DIRS_TO_PARSE_NAMES[1]):
        if x == 1:
            label = 'W'
        elif x == 8:
            label = 'B'

        if y == 4:
            label += 'K'
        elif y == 5:
            label += 'Q'

    return label
