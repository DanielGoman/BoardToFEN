from typing import Union

from src.data.consts.squares_consts import BOARD_SIDE_SIZE
from src.data.consts.piece_consts import BOARD_TO_PIECES_MAP, PIECE_TO_IGNORE


def get_piece_labels(x: int, y: int, board_type: str) -> Union[str, None]:
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
                or None if there's no need to include the requested square

    """
    label = BOARD_TO_PIECES_MAP[board_type][BOARD_SIDE_SIZE - x][y - 1]
    if label == PIECE_TO_IGNORE:
        return None

    return label
