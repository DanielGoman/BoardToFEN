from typing import Tuple
from pynput import mouse


class MouseController:
    """This class handles tracking the locations of the clicks on the screen
    Each click is registered as two, because the listener registers both the key click and the release of that key

    """
    FIRST_CLICK = 2
    SECOND_CLICK = 4

    def __init__(self):
        self.click_counter = 0
        self.point1 = None
        self.point2 = None

    def select_capture_region(self) -> Tuple[int, int, int, int]:
        """Starts the mouse key clicking listener
        This listens to every click

        Returns:
            Four integer values of top, left, bottom, right locations of the first and second click of the
            region selection

        """
        with mouse.Listener(on_click=self.on_click) as mouse_listener:
            mouse_listener.join()

        top_left = (min(self.point1[0], self.point2[0]), min(self.point1[1], self.point2[1]))
        bottom_right = (max(self.point1[0], self.point2[0]), max(self.point1[1], self.point2[1]))
        return top_left[0], top_left[1], bottom_right[0], bottom_right[1]

    def on_click(self, x, y, button, pressed):
        """Called when a mouse key is clicked or released
        When the click counter reaches self.FIRST_CLICK (which should be 2, as we register both click
        and release of the click) this method saves the location of the click.
        Same for when the counter reaches self.SECOND_CLIck, which should be 4, for the same reason as explained before

        Args:
            x: the vertical pixel index of the click
            y: the horizontal pixel index of the click
            button: the button clicked
            pressed: unused

        Returns:
            False when the second click is made, which stops the listener

        """
        if button == mouse.Button.left:
            self.click_counter += 1
            if self.click_counter == self.FIRST_CLICK:
                self.point1 = (x, y)
            elif self.click_counter == self.SECOND_CLICK:
                self.point2 = (x, y)
                return False
