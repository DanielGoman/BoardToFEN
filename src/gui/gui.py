import tkinter as tk
from typing import List, Dict


class GUI:
    def __init__(self):
        self.app = tk.Tk()

        self.image_paths = [r'C:\Users\GoMaN\Desktop\GoMaN\Projects\BoardToFEN\images\white_king.png',
                            r'C:\Users\GoMaN\Desktop\GoMaN\Projects\BoardToFEN\images\black_king.png']

        self.app.title('Board2FEN')
        self.app.geometry("300x300")

        # Create the active color button
        self.photo_images = None
        self.current_color_index = 0
        self.active_color_canvas = self.make_active_color_canvas()

        # Create En-Passant selection dropdowns
        self.square_options = {'file': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                               'row': ['1', '2', '3', '4', '5', '6', '7', '8']}
        self.file_var, self.row_var = self.make_enpassant_dropdowns(self.square_options)

        self.app.mainloop()

    def make_active_color_canvas(self):
        # Create a Canvas
        canvas = tk.Canvas(self.app, width=300, height=50)
        canvas.pack(pady=20)

        # Load images
        self.photo_images = [tk.PhotoImage(file=image_path) for image_path in self.image_paths]
        self.photo_images = [image.subsample(2) for image in self.photo_images]

        # Load the first image and create an image on the canvas
        bar_id = canvas.create_image(50, 25, anchor=tk.CENTER, image=self.photo_images[0])

        # Set up a click event to change the image
        canvas.tag_bind(bar_id, '<Button-1>', lambda event: self.change_image(canvas, bar_id))

        return canvas

    def change_image(self, canvas, bar_id):
        # Update the index to get the next image in the list
        self.current_color_index = (self.current_color_index + 1) % len(self.photo_images)

        # Update the canvas
        canvas.itemconfig(bar_id, image=self.photo_images[self.current_color_index])

    def make_enpassant_dropdowns(self, square_options: Dict[str, List[str]]):
        def reset_selections(_file_var, _row_var, _square_options):
            _file_var.set(square_options['file'][0])
            _row_var.set(square_options['row'][0])

        # Variables to store selected indices
        file_var = tk.StringVar(self.app)
        row_var = tk.StringVar(self.app)

        # Set default values
        file_var.set(square_options['file'][0])
        row_var.set(square_options['row'][0])

        # Create a Label
        label = tk.Label(self.app, text="En-passant:")
        label.pack(pady=10, padx=10, side=tk.LEFT)

        # Create the first dropdown
        dropdown1 = tk.OptionMenu(self.app, file_var, *square_options['file'])
        dropdown1.pack(pady=10, padx=10, side=tk.LEFT)

        # Create the second dropdown
        dropdown2 = tk.OptionMenu(self.app, row_var, *square_options['row'])
        dropdown2.pack(pady=10, padx=10, side=tk.LEFT)

        # Create a Reset button
        reset_button = tk.Button(self.app, text="reset",
                                 command=lambda: reset_selections(file_var, row_var, square_options))
        reset_button.pack(pady=10, padx=10, side=tk.LEFT)

        # Create a Canvas
        canvas = tk.Canvas(self.app, width=200, height=150)
        canvas.pack(pady=20)

        return file_var, row_var


if __name__ == "__main__":
    GUI()
