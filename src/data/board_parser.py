import os
import json

import numpy as np
import cv2

from typing import List, Dict, Tuple, Any

from src.board_utils.board import parse_board
from labeler import get_piece_labels
from consts.path_consts import DATA_DIR, DIRS_TO_PARSE_NAMES, PIECES_OUTPUT_DIR_PATH, LABELS_OUTPUT_FILE_PATH
from consts.squares_consts import RELEVANT_SQUARES, BOARD_SIDE_SIZE


def board_to_pieces():
    data_dirs = [DATA_DIR / data_dir for data_dir in DIRS_TO_PARSE_NAMES]

    with open(str(LABELS_OUTPUT_FILE_PATH), 'w') as labels_json:
        try:
            labels_dict = json.load(labels_json)
        except Exception as error:
            labels_dict = {}

        for i, data_dir in enumerate(data_dirs):
            for file_name in os.listdir(data_dir):
                data_dir_name = str(DIRS_TO_PARSE_NAMES[i])
                file_path = data_dir / file_name

                print(f'Parsing {str(file_path)}')

                image = cv2.imread(str(file_path))

                # TODO: check if the image has already been parsed
                # TODO: downsamples pawns/empty squares

                board_squares_dict = parse_board(image=image)

                relevant_squares_dict = get_relevant_squares(board_squares=board_squares_dict,
                                                             relevant_squares=RELEVANT_SQUARES[data_dir_name])

                labeled_squares_dict = label_squares(relevant_squares_dict, data_dir_name)

                save_squares(labeled_squares_dict=labeled_squares_dict, labels_dict=labels_dict, file_name=file_name,
                             out_json=labels_json)


def get_relevant_squares(board_squares: List[List[np.ndarray]], relevant_squares: Dict[str, List[int]]) \
        -> Dict[Tuple[int, int], np.ndarray]:
    squares_dict = {}
    for i in relevant_squares['rows']:
        for j in relevant_squares['cols']:
            squares_dict[(BOARD_SIDE_SIZE - i, j + 1)] = board_squares[(i, j)]

    return squares_dict


def label_squares(squares_dict: Dict[Tuple[int, int], np.ndarray], board_type: str) \
        -> Dict[Tuple[int, int], Dict[str, Any]]:
    labeled_squares_dict = {}
    for (i, j), square in squares_dict.items():
        square_labels = get_piece_labels(i, j, board_type)
        labeled_squares_dict[(i, j)] = {'square': square,
                                        'labels': square_labels}

    return labeled_squares_dict


def save_squares(labeled_squares_dict: Dict[Tuple[int, int], np.ndarray], labels_dict: dict, file_name: str,
                 out_json: str):
    for (i, j), square_data in labeled_squares_dict.items():
        stripped_file_name = '.'.join(file_name.split('.')[:-1])
        out_file_name = f'{stripped_file_name}_{i}_{j}'
        full_out_file_name = f'{out_file_name}.png'
        out_file_path = PIECES_OUTPUT_DIR_PATH / full_out_file_name

        print(f'Writing {str(full_out_file_name)}')
        cv2.imwrite(str(out_file_path), square_data['square'])

        if stripped_file_name not in labels_dict.keys():
            labels_dict[stripped_file_name] = {}
        labels_dict[stripped_file_name][str((i, j))] = {'piece_type': square_data['labels'][0],
                                                        'piece_color': square_data['labels'][1],
                                                        'image_file_name': full_out_file_name}

    json.dump(labels_dict, out_json, indent=4, sort_keys=True)


if __name__ == "__main__":
    board_to_pieces()
