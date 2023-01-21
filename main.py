from src.input_utils.keyboard_controler import KeyboardController
from src.board_utils.board import Board


def main():
    board_template_path = r'src/data/empty_board_template.png'
    board = Board(board_template_path=board_template_path)

    controller = KeyboardController(board=board)
    controller.start_listener()



if __name__ == "__main__":
    main()
