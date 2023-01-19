import os
from typing import List, Dict, Tuple

import numpy as np
import cv2

from consts import DATA_DIR, DIRS_TO_PARSE_NAMES, RELEVANT_SQUARES, OUTPUT_DIR_PATH, BOARD_SIDE_SIZE
from src.board_utils.board import parse_board


def main():
    data_dirs = [DATA_DIR / data_dir for data_dir in DIRS_TO_PARSE_NAMES]

    for i, data_dir in enumerate(data_dirs):
        for file_name in os.listdir(data_dir):
            data_dir_name = str(DIRS_TO_PARSE_NAMES[i])
            file_path = data_dir / file_name

            print(f'Parsing {str(file_path)}')

            image = cv2.imread(str(file_path))

            # TODO: check if the image has already been parsed
            # TODO: this needs to be improved, need to also include empty squares
            # TODO: no need to have many copies of pawns, remove 6 of each color (keep one on white
            #           square and one on a black square)
            # TODO: save piece labels

            board_squares_dict = parse_board(image=image)

            relevant_squares_dict = get_relevant_squares(board_squares=board_squares_dict,
                                                         relevant_squares=RELEVANT_SQUARES[data_dir_name])

            save_squares(squares_dict=relevant_squares_dict, file_name=file_name)


def get_relevant_squares(board_squares: List[List[np.ndarray]], relevant_squares: Dict[str, List[int]]) \
        -> Dict[Tuple[int, int], np.ndarray]:
    squares_dict = {}
    for i in relevant_squares['rows']:
        for j in relevant_squares['cols']:
            squares_dict[(BOARD_SIDE_SIZE - i, j + 1)] = board_squares[(i, j)]

    return squares_dict


def save_squares(squares_dict: Dict[Tuple[int, int], np.ndarray], file_name):
    for (i, j), square in squares_dict.items():
        stripped_file_name = '.'.join(file_name.split('.')[:-1])
        out_file_name = f'{stripped_file_name}_{i}_{j}.png'
        out_file_path = OUTPUT_DIR_PATH / out_file_name

        print(f'Writing {str(out_file_name)}')
        cv2.imwrite(str(out_file_path), square)


if __name__ == "__main__":
    main()
