import numpy as np

from typing import Tuple
from PIL import ImageGrab

from src.input_utils.mouse_controller import MouseController


class ImageCapture:
    """This class handles the screenshot capturing mechanism

    """
    def __init__(self):
        self.controller = MouseController()

    def capture(self) -> np.ndarray:
        """Captures a selected frame of the screen and processes that frame

        Returns:
            out_frame: the captured frame as an array

        """
        capture_region = self.controller.select_capture_region()
        frame = ImageCapture.capture_image(capture_region)
        bgr_frame = frame[..., ::-1]
        out_frame = np.ascontiguousarray(bgr_frame, np.uint8)

        return out_frame

    @staticmethod
    def capture_image(capture_region: Tuple[int, int, int, int]) -> np.ndarray:
        """Captures a selected frame of the screen

        Args:
            capture_region: top, left, bottom, right location on the screen to capture

        Returns:
            frame: the captured frame as an array

        """
        capture = ImageGrab.grab(bbox=capture_region)
        frame = np.array(capture)

        return frame


if __name__ == "__main__":
    cap = ImageCapture()
    img = cap.capture()
