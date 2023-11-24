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
        self.top_left = None
        self.bottom_right = None

    def select_capture_region(self) -> Tuple[int, int, int, int]:
        """Starts the mouse key clicking listener
        This listens to every click

        Returns:
            Four integer values of top, left, bottom, right locations of the first and second click of the
            region selection

        """
        with mouse.Listener(on_click=self.on_click) as mouse_listener:
            mouse_listener.join()

        return *self.top_left, *self.bottom_right

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
                self.top_left = (x, y)
            elif self.click_counter == self.SECOND_CLICK:
                self.bottom_right = (x, y)
                return False


if __name__ == "__main__":
    controller = MouseController()
    controller.select_capture_region()
