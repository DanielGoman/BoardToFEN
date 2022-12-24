from pynput import keyboard
from src.input_utils.image_capture import ImageCapture
from src.board_utils.board import Board


class KeyboardController:
    # To start capturing a single frame, click c
    def __init__(self, board: Board, start_key: str = 'c'):
        self.start_key = start_key
        self.board = board

    def start_listener(self):
        with keyboard.Listener(on_press=self.on_press) as keyboard_listener:
            keyboard_listener.join()

    def on_press(self, key):
        if key.char == self.start_key:
            cap = ImageCapture()
            img = cap.capture()

            self.board.convert_to_fen(frame=img)


if __name__ == "__main__":
    controller = KeyboardController()
    controller.start_listener()
