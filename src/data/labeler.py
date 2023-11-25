from consts.path_consts import DIRS_TO_PARSE_NAMES


def get_piece_labels(x: int, y: int, board_type: str) -> str:
    """Converts the square indices into the piece that is there.
    This assumes one of two input image types - either full boards, or boards that have only a king and a queen
    with replaced locations.
    Hence, for every type of input board, we know the piece that is supposed to be at any square, and we label them
    accordingly

    Args:
        x: row index
        y: column index
        board_type: board type, according to the `DIRS_TO_PARSE_NAMES` const

    Returns:
        label: the proper label to the given square w.r.t the type of the board (board_type)

    """
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
