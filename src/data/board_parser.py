import os
import json

import cv2
import numpy as np

from typing import List, Dict, Tuple, Any

from src.board.board import parse_board
from labeler import get_piece_labels
from consts.path_consts import DATA_DIR, DIRS_TO_PARSE_NAMES, PIECES_OUTPUT_DIR_PATH, LABELS_OUTPUT_FILE_PATH
from consts.squares_consts import RELEVANT_SQUARES, BOARD_SIDE_SIZE


def board_to_pieces():
    """Takes in directories of board images.
    Makes a new directory of images of selected squares out of all those boards.
    Makes a new `.json` file that contains the respective labels

    The structure of the `.json` labels file is as follows:
    {
        'board_type':{
            'image_id':{
                '(square_x, square_y)':
                {
                    'image_file_name': <image name>,
                    'label': <label>
                }
            }
        }
    }

    """
    data_dirs = [DATA_DIR / data_dir for data_dir in DIRS_TO_PARSE_NAMES]

    with open(str(LABELS_OUTPUT_FILE_PATH), 'w') as labels_json:
        try:
            labels_dict = json.load(labels_json)
        except Exception as error:
            labels_dict = {}

        for i, data_dir in enumerate(data_dirs):
            data_dir_name = str(DIRS_TO_PARSE_NAMES[i])
            if data_dir_name not in labels_dict:
                labels_dict[data_dir_name] = {}

            for file_name in os.listdir(data_dir):
                file_path = data_dir / file_name
                stripped_file_name = '.'.join(file_name.split('.')[:-1])

                print(f'Parsing {str(file_path)}')

                image = cv2.imread(str(file_path))

                board_squares_dict = parse_board(image=image)

                relevant_squares_dict = get_relevant_squares(board_squares=board_squares_dict,
                                                             relevant_squares=RELEVANT_SQUARES[data_dir_name])

                labeled_squares_dict = label_squares(relevant_squares_dict, data_dir_name)

                labels_dict[data_dir_name][stripped_file_name] = save_squares(labeled_squares_dict=labeled_squares_dict,
                                                                              data_dir_index=i,
                                                                              stripped_file_name=stripped_file_name)

        json.dump(labels_dict, labels_json, indent=4, sort_keys=True)


def get_relevant_squares(board_squares: Dict[Tuple[int, int], np.ndarray], relevant_squares: Dict[str, List[int]]) \
        -> Dict[Tuple[int, int], np.ndarray]:
    """Takes only the relevant squares from the board, w.r.t to the type of the board

    Args:
        board_squares: "2d" dict of squares.
                        board_squares[(square_x, square_y)] = np.ndarray of the square
        relevant_squares: the rows and columns to select out of the entire board

    Returns:
        squares_dict: dict similar to board_squares, of only the selected squares

    """
    squares_dict = {}
    for i in relevant_squares['rows']:
        for j in relevant_squares['cols']:
            squares_dict[(BOARD_SIDE_SIZE - i, j + 1)] = board_squares[(i, j)]

    return squares_dict


def label_squares(squares_dict: Dict[Tuple[int, int], np.ndarray], board_type: str) \
        -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Gives the proper label to each piece on the board, w.r.t the board type

    Args:
        squares_dict: "2d" dict of squares.
                        board_squares[(square_x, square_y)] = np.ndarray of the square
        board_type: the type of the board - either full board or a board with only the kings and the queens with their
                        locations replaced

    Returns:
        labeled_squares_dict: dict for every selected square, of the square's image and its respective label

    """
    labeled_squares_dict = {}
    for (i, j), square in squares_dict.items():
        square_label = get_piece_labels(i, j, board_type)
        labeled_squares_dict[(i, j)] = {'square': square,
                                        'label': square_label}

    return labeled_squares_dict


def save_squares(labeled_squares_dict: Dict[Tuple[int, int], Dict[str, Any]], data_dir_index: int,
                 stripped_file_name: str) -> Dict[Tuple[int, int], Dict[str, str]]:
    """Saves the squares images as a `.png` file, and puts their respective labels in a dict that will be later saved
    as a `.json` file

    Args:
        labeled_squares_dict: dict for every selected square, of the square's image and its respective label
        data_dir_index: the index that represents to which board type the current dict belongs to
        stripped_file_name: the file name of the board image for the current board type

    Returns:
        labels_dict: dict for every selected square, of the square's label and the name of the board image it belong to

    """
    labels_dict = {}
    for (i, j), square_data in labeled_squares_dict.items():
        out_file_name = f'{data_dir_index}_{stripped_file_name}_{i}_{j}.png'
        out_file_path = PIECES_OUTPUT_DIR_PATH / out_file_name

        print(f'Writing {str(out_file_name)}')
        cv2.imwrite(str(out_file_path), square_data['square'])

        labels_dict[str((i, j))] = {'label': square_data['label'],
                                    'image_file_name': out_file_name}

    return labels_dict


if __name__ == "__main__":
    board_to_pieces()
