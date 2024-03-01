from typing import List

import tkinter as tk
from tkinter import BooleanVar, StringVar

from src.app.consts import ActiveColor
from src.fen_converter.fen_converter import convert_board_pieces_to_fen, convert_fen_to_url


def gui_to_fen_parameters(board_rows_as_fen: List[str], current_color_index: int,
                          castling_rights_checkboxes: List[BooleanVar], file_var: StringVar, row_var: StringVar,
                          n_halfmoves_var: StringVar, n_fullmoves_var: StringVar, domain_number: int,
                          default_value: str) -> str:
    """Takes all parameters from the GUI and combines them with the already generated partial FEN from the screenshot
    to create a full FEN for the requested domain

    Args:
        board_rows_as_fen: list of strings - one per row, in FEN format
        current_color_index: index of the currently selected active color
        castling_rights_checkboxes: vars containing the status of each castling rights button
        file_var: variable containing the currently specified en-passant file
        row_var: variable containing the currently specified en-passant row
        n_halfmoves_var: variable containing the currently specified number of the halfmoves dropdown
        n_fullmoves_var: variable contains the currently specified number of full moves
        domain_number: the number of domain to generate FEN for
        default_value: default value in dropdowns

    Returns:
        fen_url: full FEN url for the requested domain

    """
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


def show_disappearing_message_box(app: tk.Tk, message: str):
    """Displays a disappearing message box

    Args:
        app: the app object to on which we display the message box
        message: content of the message to be displayed

    """
    message_box = tk.Toplevel(app)
    width = app.winfo_width()
    height = 20
    x = app.winfo_x() +app.winfo_width() // 2 - width // 2
    y = app.winfo_y() + 35
    message_box.geometry(f"{width}x{height}+{x}+{y}")
    message_box.overrideredirect(True)
    message_box.attributes('-topmost', True)

    message_label = tk.Label(message_box, text=message, fg='red', bg='black', bd=4, font=('Arial', 12, 'bold'))
    message_label.pack()
    app.after(3000, message_box.destroy)
