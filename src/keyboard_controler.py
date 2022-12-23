from matplotlib import pyplot as plt
from pynput import keyboard
from image_capture import ImageCapture


class KeyboardController:
    # To start capturing a single frame, click c
    def __init__(self, start_key: str = 'c'):
        self.start_key = start_key

    def start_listener(self):
        with keyboard.Listener(on_press=self.on_press) as keyboard_listener:
            keyboard_listener.join()

    def on_press(self, key):
        if key.char == self.start_key:
            cap = ImageCapture()
            plt.imshow(cap.capture())
            plt.show()
            # TODO: add logic


if __name__ == "__main__":
    controller = KeyboardController()
    controller.start_listener()