import tkinter as tk
from tkinter import ttk
from typing import List, Dict


class GUI:
    def __init__(self):
        self.app = tk.Tk()

        self.active_color_image_paths = [r'C:\Users\GoMaN\Desktop\GoMaN\Projects\BoardToFEN\images\white_king.png',
                                         r'C:\Users\GoMaN\Desktop\GoMaN\Projects\BoardToFEN\images\black_king.png']
        self.screenshot_image_path = r'C:\Users\GoMaN\Desktop\GoMaN\Projects\BoardToFEN\images\screenshot.png'

        self.app.title('Board2FEN')
        self.app.geometry("300x300")

        # Create the active color button
        self.photo_images = None
        self.current_color_index = 0
        self.active_color_canvas = self.make_active_color_canvas()

        # Create En-Passant selection dropdowns
        self.dropdown_default_value = '-'
        self.square_options = {'file': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                               'row': ['1', '2', '3', '4', '5', '6', '7', '8']}
        self.file_var, self.row_var = self.make_enpassant_dropdowns(self.square_options)

        # Create half-moves dropdown
        self.halfmove_options = list(range(50))
        self.n_halfmoves_var = self.make_halfmoves_dropdown(self.halfmove_options)

        # Create full-move entry
        self.default_fullmoves = 0
        self.n_fullmoves_var = self.make_fullmoves_entry()

        self.app.mainloop()

    def make_active_color_canvas(self):
        # Create a Canvas
        canvas = tk.Canvas(self.app, width=300, height=50)
        canvas.pack(pady=20)

        # Load images
        self.photo_images = [tk.PhotoImage(file=image_path) for image_path in self.active_color_image_paths]
        self.photo_images = [image.subsample(2) for image in self.photo_images]

        # Load the first image and create an image on the canvas
        bar_id = canvas.create_image(50, 25, anchor=tk.CENTER, image=self.photo_images[0])

        # Load screenshot image
        screenshot_image = tk.PhotoImage(file=self.screenshot_image_path)
        screenshot_image = screenshot_image.subsample(6)

        # Make screenshot click popup message
        popup_label = tk.Label(self.app, text="", bg="white", relief=tk.SOLID, borderwidth=1)
        popup_label.place_forget()

        # Make screenshot button
        button = tk.Button(self.app, image=screenshot_image, command=lambda: self.on_screenshot_click(popup_label))
        button.image = screenshot_image
        button.place(x=220, y=25)  # Adjust the position as needed

        # Set up a click event to change the image
        canvas.tag_bind(bar_id, '<Button-1>', lambda event: self.change_image(canvas, bar_id))

        return canvas

    def change_image(self, canvas, bar_id):
        # Update the index to get the next image in the list
        self.current_color_index = (self.current_color_index + 1) % len(self.photo_images)

        # Update the canvas
        canvas.itemconfig(bar_id, image=self.photo_images[self.current_color_index])

    def show_popup(self, popup_label):
        message = 'Take a screenshot!'
        popup_label.config(text=message)
        popup_label.place(x=(self.app.winfo_reqwidth() - popup_label.winfo_reqwidth()) // 2, y=10)

        #  Hide the popup after 2000 milliseconds (2 seconds)
        self.app.after(2000, lambda: popup_label.place_forget())

    def on_screenshot_click(self, popup_label):
        self.show_popup(popup_label)
        pass

    def make_enpassant_dropdowns(self, square_options: Dict[str, List[str]]):
        def reset_selections(_file_var, _row_var):
            _file_var.set(self.dropdown_default_value)
            _row_var.set(self.dropdown_default_value)

        # Create a Canvas
        canvas = tk.Canvas(self.app, width=250, height=25)
        canvas.pack()

        # Variables to store selected indices
        file_var = tk.StringVar(self.app)
        row_var = tk.StringVar(self.app)

        # Set default values
        file_var.set(self.dropdown_default_value)
        row_var.set(self.dropdown_default_value)

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
        def reset_selections(_halfmoves_var):
            _halfmoves_var.set(self.dropdown_default_value)

        # Create a Canvas
        canvas = tk.Canvas(self.app, width=250, height=10)
        canvas.pack(pady=10)

        # Variables to store selected indices
        halfmoves_var = tk.StringVar(self.app)

        # Set default values
        halfmoves_var.set(self.dropdown_default_value)

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

        entry_var = tk.StringVar()
        entry_var.set(str(self.default_fullmoves))

        # Entry widget with validation
        entry = tk.Entry(self.app, textvariable=entry_var, validate="key",
                         validatecommand=(self.app.register(validate_input), '%S'))

        # Create a Reset button
        reset_button = tk.Button(self.app, text="reset",
                                 command=lambda: reset_entry(entry_var))
        reset_button.pack(padx=10)

        canvas.create_window(25, 10, window=label)
        canvas.create_window(130, 10, window=entry)
        canvas.create_window(230, 10, window=reset_button)

        return entry_var


if __name__ == "__main__":
    GUI()
