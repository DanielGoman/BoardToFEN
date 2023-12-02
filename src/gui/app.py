import tkinter as tk

from tkinter import ttk
from typing import List, Dict, Tuple

from src.fen_converter.fen_converter import convert_board_pieces_to_fen, convert_fen_to_url
from src.gui.app_utils import gui_to_fen_parameters
from src.gui.consts import ActiveColor
from src.pipeline.pipeline import Pipeline


class App:
    """This class manages the interactions of the user with the software.
    A simple GUI is displayed for the user, that allow them to screenshot their board and convert it to a FEN,
    including various board parameters that can't be deduced simply from the board position, such as:
        - Active color
        - Castling rights
        - Available en-passant
        - Number of half moves (number of moves made since the last pawn move/piece capture, capped at 50)
        - Number of full moves (number of moves made in the game)

    """

    def __init__(self, pipeline: Pipeline, active_color_image_paths: Dict[str, str],
                 screenshot_image_path: str, domain_logo_paths: Dict[int, str]):
        self.pipeline = pipeline
        self.board_rows_as_fen = None

        self.app = tk.Tk()
        self.app.title('Board2FEN')
        self.app.geometry("300x400")

        # Create the active color button
        self.photo_images = None
        self.current_color_index = 0
        self.make_active_color_button_and_screen_shot_button(active_color_image_paths, screenshot_image_path)

        # Create castling rights checkboxes
        self.checkbox_texts = ['White king-side castle', 'White Queen-side castle',
                               'Black king-side castle', 'Black Queen-side castle']
        self.castling_rights_checkboxes = self.make_castling_availability_checkboxes(self.checkbox_texts)

        # Create En-Passant selection dropdowns
        self.default_value = '-'
        self.square_options = {'file': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                               'row': ['1', '2', '3', '4', '5', '6', '7', '8']}
        self.file_var, self.row_var = self.make_enpassant_dropdowns(self.square_options)

        # Create half-moves dropdown
        self.halfmove_options = list(range(50))
        self.n_halfmoves_var = self.make_halfmoves_dropdown(self.halfmove_options)

        # Create full-move entry
        self.default_fullmoves = 0
        self.n_fullmoves_var = self.make_fullmoves_entry()

        self.make_domain_buttons(domain_logo_paths)

    def start_app(self):
        self.app.mainloop()

    def make_active_color_button_and_screen_shot_button(self, active_color_image_paths: Dict[str, str],
                                                        screenshot_image_path: str):
        """Creates the button which accepts from the user the current active color, as well as the screenshot button

        Args:
            active_color_image_paths: paths to images of each color
            screenshot_image_path: path to the image of the screenshot button

        """
        # Create a Canvas
        canvas = tk.Canvas(self.app, width=300, height=50)
        canvas.pack(pady=(20, 5))

        # Load images
        self.photo_images = [tk.PhotoImage(file=image_path) for image_path in active_color_image_paths.values()]
        self.photo_images = [image.subsample(2) for image in self.photo_images]

        # Load the first image and create an image on the canvas
        bar_id = canvas.create_image(50, 25, anchor=tk.CENTER, image=self.photo_images[0])

        # Load screenshot image
        screenshot_image = tk.PhotoImage(file=screenshot_image_path)
        screenshot_image = screenshot_image.subsample(6)

        # Make screenshot click popup message
        popup_label = tk.Label(self.app, text="", bg="white", relief=tk.SOLID, borderwidth=1)
        popup_label.place_forget()

        # Make screenshot button
        button = tk.Button(self.app, image=screenshot_image, command=self.on_screenshot_click)
        button.image = screenshot_image
        button.place(x=220, y=25)  # Adjust the position as needed

        # Set up a click event to change the image
        canvas.tag_bind(bar_id, '<Button-1>', lambda event: self.change_image(canvas, bar_id))

        return canvas

    def change_image(self, canvas: tk.Canvas, bar_id: int):
        """Changes the image of the active color button when clicked

        Args:
            canvas: canvas on which the image is displayed
            bar_id: id of the button

        """
        # Update the index to get the next image in the list
        self.current_color_index = (self.current_color_index + 1) % len(self.photo_images)

        # Update the canvas
        canvas.itemconfig(bar_id, image=self.photo_images[self.current_color_index])

    def make_castling_availability_checkboxes(self, checkbox_texts: List[str]) -> List[tk.BooleanVar]:
        """Creates four castling rights checkboxes:
            - White king-side
            - White queen-sie
            - Black king-side
            - Black queen-side

        Args:
            checkbox_texts: the strings displayed along each checkbox

        Returns:
            castling_rights_checkboxes: vars containing the status of each castling rights button

        """
        castling_rights_checkboxes = []
        for i in range(4):
            frame = tk.Frame(self.app)
            frame.pack(side=tk.TOP, padx=5)  # Pack frames along the top

            # Checkbutton
            check_var = tk.BooleanVar()
            checkbutton = tk.Checkbutton(frame, text=checkbox_texts[i], variable=check_var)
            checkbutton.pack(side=tk.LEFT)
            castling_rights_checkboxes.append(check_var)

        return castling_rights_checkboxes

    def make_enpassant_dropdowns(self, square_options: Dict[str, List[str]]) -> Tuple[tk.StringVar, tk.StringVar]:
        """Creates en-passant dropdowns, one for the rows, and one for the columns

        Args:
            square_options: dict of available options to display per dropdown

        Returns:
            file_var: variable containing the currently specified en-passant file
            row_var: variable containing the currently specified en-passant row

        """
        def reset_selections(_file_var, _row_var):
            _file_var.set(self.default_value)
            _row_var.set(self.default_value)

        # Create a Canvas
        canvas = tk.Canvas(self.app, width=250, height=25)
        canvas.pack(pady=(10, 0))

        # Variables to store selected indices
        file_var = tk.StringVar(self.app)
        row_var = tk.StringVar(self.app)

        # Set default values
        file_var.set(self.default_value)
        row_var.set(self.default_value)

        # Create a Label
        label = tk.Label(self.app, text="En-passant:")
        label.pack(padx=10)

        # Create the first dropdown
        file_dropdown = tk.OptionMenu(self.app, file_var, *square_options['file'])
        file_dropdown.pack(pady=10, padx=10)

        # Create the second dropdown
        row_dropdown = tk.OptionMenu(self.app, row_var, *square_options['row'])
        row_dropdown.pack(pady=10, padx=10)

        # Create a Reset button
        reset_button = tk.Button(self.app, text="reset",
                                 command=lambda: reset_selections(file_var, row_var))
        reset_button.pack(pady=10, padx=10)

        canvas.create_window(25, 10, window=label)
        canvas.create_window(100, 10, window=file_dropdown)
        canvas.create_window(160, 10, window=row_dropdown)
        canvas.create_window(230, 10, window=reset_button)

        return file_var, row_var

    def make_halfmoves_dropdown(self, halfmove_options: List[int]):
        """Creates the half moves dropdown

        Args:
            halfmove_options: list of available options to display in the dropdown

        Returns:
            halfmoves_var: variable containing the currently specified number of the halfmoves dropdown

        """
        def reset_selections(_halfmoves_var):
            _halfmoves_var.set(self.default_value)

        # Create a Canvas
        canvas = tk.Canvas(self.app, width=250, height=10)
        canvas.pack(pady=10)

        # Variables to store selected indices
        halfmoves_var = tk.StringVar(self.app)

        # Set default values
        halfmoves_var.set(self.default_value)

        # Create a Label
        label = tk.Label(self.app, text="Half-moves:")

        # Create the first dropdown
        halfmoves_combobox = ttk.Combobox(self.app, textvariable=halfmoves_var, values=halfmove_options, height=5,
                                          width=5, state="readonly")
        halfmoves_combobox.pack(padx=10)

        # Create a Reset button
        reset_button = tk.Button(self.app, text="reset",
                                 command=lambda: reset_selections(halfmoves_var))
        reset_button.pack(padx=10)

        canvas.create_window(25, 10, window=label)
        canvas.create_window(130, 10, window=halfmoves_combobox)
        canvas.create_window(230, 10, window=reset_button)

        return halfmoves_var

    def make_fullmoves_entry(self):
        """Creates a full moves entry (digits only)

        Returns:
            fullmoves_entry_var: variable contains the currently specified number of full moves

        """
        def validate_input(char):
            # This function is called whenever a key is pressed
            # It checks if the input is a digit or an empty string (allowing deletion)
            return char.isdigit() or char == ""

        def reset_entry(_entry_var):
            _entry_var.set(str(self.default_fullmoves))

        canvas = tk.Canvas(self.app, width=250, height=10)
        canvas.pack(pady=10)

        # Label to display text to the left of the Entry widget
        label = tk.Label(self.app, text="Full-moves:")

        fullmoves_entry_var = tk.StringVar()
        fullmoves_entry_var.set(str(self.default_fullmoves))

        # Entry widget with validation
        entry = tk.Entry(self.app, textvariable=fullmoves_entry_var, validate="key",
                         validatecommand=(self.app.register(validate_input), '%S'))

        # Create a Reset button
        reset_button = tk.Button(self.app, text="reset",
                                 command=lambda: reset_entry(fullmoves_entry_var))
        reset_button.pack(padx=10)

        canvas.create_window(25, 10, window=label)
        canvas.create_window(130, 10, window=entry)
        canvas.create_window(230, 10, window=reset_button)

        return fullmoves_entry_var

    def make_domain_buttons(self, image_paths: Dict[int, str]):
        """Creates the domain buttons

        Args:
            image_paths: paths to images of each domain

        """
        domain_keys = list(image_paths.keys())
        image_paths = list(image_paths.values())

        # PhotoImage instances for the buttons
        image_1 = tk.PhotoImage(file=image_paths[0]).subsample(2)
        image_2 = tk.PhotoImage(file=image_paths[1]).subsample(2)
        image_3 = tk.PhotoImage(file=image_paths[2]).subsample(2)

        # Function to create a button with an image
        def create_image_button(image, button_number):
            button = tk.Button(self.app, image=image, command=lambda: self.on_domain_click(button_number))
            button.image = image
            button.pack(side=tk.LEFT, padx=(30, 30), pady=5)

        # Create three buttons with images
        create_image_button(image_1, domain_keys[0])
        create_image_button(image_2, domain_keys[1])
        create_image_button(image_3, domain_keys[2])

    def on_screenshot_click(self):
        """Lets the user take a screenshot, and converts the board in the screenshot to FEN format

        """
        self.app.iconify()
        self.board_rows_as_fen = self.pipeline.run_pipeline()
        self.app.deiconify()

    def on_domain_click(self, domain_number: int):
        """Converts FEN parameters to a full fen w.r.t the type of domain required

        Args:
            domain_number: the type of domain to create the fen for

        """
        fen_url = gui_to_fen_parameters(board_rows_as_fen=self.board_rows_as_fen,
                                        current_color_index=self.current_color_index,
                                        castling_rights_checkboxes=self.castling_rights_checkboxes,
                                        file_var=self.file_var,
                                        row_var=self.row_var, default_value=self.default_value,
                                        n_halfmoves_var=self.n_halfmoves_var, n_fullmoves_var=self.n_fullmoves_var,
                                        domain_number=domain_number)

        print(fen_url)
        # ?
