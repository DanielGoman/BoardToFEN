import numpy as np

from PIL import ImageGrab


class ImageCapture:
    def __init__(self):
        self.top_left = None
        self.bottom_right = None

    def capture(self):
        pass

    def select_capture_region(self):
        pass

    def capture_image(self,) -> np.ndarray:
        capture_region = (*self.top_left, *self.bottom_right)

        capture = ImageGrab.grab(bbox=capture_region)
        frame = np.array(capture)

        return frame


if __name__ == "__main__":
    cap = ImageCapture()
    cap.top_left = (50, 50)
    cap.bottom_right = (700, 700)

    from matplotlib import pyplot as plt
    plt.imshow(cap.capture_image())
    plt.show()
