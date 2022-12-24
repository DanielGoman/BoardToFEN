from typing import Tuple

import numpy as np
from PIL import ImageGrab
from src.input_utils.mouse_controller import MouseController


class ImageCapture:
    def __init__(self):
        self.controller = MouseController()

    def capture(self):
        capture_region = self.controller.select_capture_region()
        frame = ImageCapture.capture_image(capture_region)
        bgr_frame = frame[..., ::-1]
        out_frame = np.ascontiguousarray(bgr_frame, np.uint8)

        return out_frame

    @staticmethod
    def capture_image(capture_region: Tuple[int, int, int, int]) -> np.ndarray:
        capture = ImageGrab.grab(bbox=capture_region)
        frame = np.array(capture)

        return frame


if __name__ == "__main__":
    cap = ImageCapture()

    img = cap.capture()

