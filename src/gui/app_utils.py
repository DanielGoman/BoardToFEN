from tkinter import BooleanVar, StringVar
from typing import List

from src.fen_converter.fen_converter import convert_board_pieces_to_fen, convert_fen_to_url
from src.gui.consts import ActiveColor


def gui_to_fen_parameters(board_rows_as_fen: List[str], current_color_index: int,
                          castling_rights_checkboxes: List[BooleanVar], file_var: StringVar, row_var: StringVar,
                          n_halfmoves_var: StringVar, n_fullmoves_var: StringVar, domain_number: int,
                          default_value: str) -> str:
    active_color = current_color_index == ActiveColor.White.value

    castling_rights = [bool(checkbox_var.get()) for checkbox_var in castling_rights_checkboxes]

    if file_var.get() != default_value and row_var != default_value:
        possible_en_passant = file_var.get() + row_var.get()
    else:
        possible_en_passant = None

    n_half_moves = n_halfmoves_var.get() if n_halfmoves_var.get() != default_value else 0
    n_full_moves = n_fullmoves_var.get()

    fen_parts = convert_board_pieces_to_fen(active_color=active_color, castling_rights=castling_rights,
                                            possible_en_passant=possible_en_passant, n_half_moves=n_half_moves,
                                            n_full_moves=n_full_moves)
    fen_parts = [board_rows_as_fen, *fen_parts]
    fen_url = convert_fen_to_url(fen_parts=fen_parts, domain=domain_number)

    return fen_url
