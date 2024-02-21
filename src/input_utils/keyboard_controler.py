from pynput import keyboard
from pynput.keyboard import KeyCode

from src.pipeline.pipeline import Pipeline


class KeyboardController:
    """The screenshot capture controller, triggered by a key press
    To start capturing a single frame, click c

    """
    def __init__(self, pipeline: Pipeline, start_key: str = 'c'):
        self.start_key = start_key
        self.pipeline = pipeline

    def start_listener(self):
        """Starts the key pressing listener
        This listens to every click

        """
        with keyboard.Listener(on_press=self.on_press) as keyboard_listener:
            keyboard_listener.join()

    def on_press(self, key):
        """Called when a keyboard key is pressed
        Captures a partial screenshot of the screen and follows up with detection of the chess board in the captured
        frame, as well as piece recognition

        Args:
            key: the keyboard key pressed

        """
        if key == KeyCode.from_char(self.start_key):
            self.pipeline.run_pipeline()


if __name__ == "__main__":
    controller = KeyboardController()
    controller.start_listener()
