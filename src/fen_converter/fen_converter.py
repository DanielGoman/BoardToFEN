import numpy as np

from typing import List, Union

from src.data.consts.squares_consts import BOARD_SIDE_SIZE
from src.fen_converter.fen_utils import convert_row_to_fen
from src.fen_converter.consts import ACTIVE_COLOR_MAPPING, CASTLING_RIGHTS_MAPPING, FenPartDefaults


def convert_board_pieces_to_fen(pieces: np.ndarray, active_color: bool, castling_rights: List[bool],
                                possible_en_passant: Union[str, None], n_half_moves: Union[int, None],
                                n_full_moves: Union[int, None]) \
        -> List[str]:
    """Converts the predicted pieces, in addition to the other FEN-relevant parameters into a FEN
    This is implemented according to the following chess.com article:
    https://www.chess.com/terms/fen-chess

    Args:
        pieces: 1d array of the pieces as their numerical representation
                (can be found in src/data/consts/pieces_consts.py)
                Starts from the upper left corner of the board and moves right and downwards
        active_color: the player whose turn to move [w, b]
        castling_rights: list of legally available castling options
                            [Q, q] - queen-side castle is available for white/black respectively
                            [K, k] - king-side castle is available for white/black respectively
        possible_en_passant: string of the square that can be captured as en-passant on this turn, None if not possible
        n_half_moves: number of moves since the last pawn move or the last piece capture
        n_full_moves: move counter

    Returns:
        fen_parts: list of the parts of the FEN format

    """
    fen_parts = []

    # Part 1 - Board pieces
    piece_rows = list(pieces.reshape((BOARD_SIDE_SIZE, BOARD_SIDE_SIZE)))
    board_rows_as_fen = [convert_row_to_fen(row) for row in piece_rows]
    fen_parts.append(board_rows_as_fen)

    # Part 2 - Active color
    fen_parts.append(ACTIVE_COLOR_MAPPING[active_color])

    # Part 3 - Possible castling right
    available_castling_rights_indices = list(np.where(castling_rights))
    available_castling_rights_strings = [CASTLING_RIGHTS_MAPPING[idx] for idx in available_castling_rights_indices]
    available_castling_rights = ''.join(available_castling_rights_strings) if available_castling_rights_strings \
        else FenPartDefaults.str.value
    fen_parts.append(available_castling_rights)

    # Part 4 - Possible en passant
    if not possible_en_passant:
        possible_en_passant = FenPartDefaults.str.value
    fen_parts.append(possible_en_passant)

    # Part 5 - Number of moves since last pawn move / piece capture
    if not n_half_moves:
        n_half_moves = FenPartDefaults.int.value
    fen_parts.append(n_half_moves)

    # Part 6 - Number of moves
    if not n_full_moves:
        n_full_moves = FenPartDefaults.int.value
    fen_parts.append(n_full_moves)

    return fen_parts


def convert_fen_to_url(fen_parts: List[str], domain: str) -> str:
    """Joins the parts of the FEN format into a full url to the specified domain, according to the formatting each
    domain requires

    Args:
        fen_parts: list of the parts of the FEN format
        domain: the domain that the FEN link is generated for [chess.com, lichess.org]

    Returns:
        @: full fen url to the specified domain

    """
    pass
