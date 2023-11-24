from src.input_utils.keyboard_controler import KeyboardController
from src.board_utils.board import Board


def main():
    controller = KeyboardController()
    controller.start_listener()


if __name__ == "__main__":
    main()
