
from pynput import mouse


class MouseController:
    FIRST_CLICK = 2
    SECOND_CLICK = 4

    def __init__(self):
        self.click_counter = 0
        self.top_left = None
        self.bottom_right = None

    def select_capture_region(self):
        with mouse.Listener(on_click=self.on_click) as listener:
            listener.join()
            print(self.top_left)
            print(self.bottom_right)
        print(self.top_left)
        print(self.bottom_right)

    def on_click(self, x, y, button, pressed):
        if button == mouse.Button.left:
            print('Clicked')
            self.click_counter += 1
            if self.click_counter == self.FIRST_CLICK:
                print('first')
                self.top_left = (x, y)
            elif self.click_counter == self.SECOND_CLICK:
                print('second')
                self.bottom_right = (x, y)
                return False


if __name__ == "__main__":
    controller = MouseController()
    controller.select_capture_region()
