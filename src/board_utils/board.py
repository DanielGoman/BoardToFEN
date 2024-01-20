from __future__ import annotations

import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from typing import Dict, Tuple, Optional

from src.board_utils.consts import Canny
from src.board_utils.old_board import get_edge_image, get_max_seq_lens_per_row, split_board_image_to_squares, crop_image

logger = logging.getLogger(__name__)


class Board:
    N_LINES = 9

    def __init__(self, board: np.ndarray):
        self.board = board

    @property
    def area(self) -> int:
        return self.board.shape[0] * self.board.shape[1]

    def __gt__(self, other: Board) -> bool:
        return self.area > other.area

    def split_board_into_squares(self) -> Dict[Tuple[int, int], np.ndarray]:
        board_height = self.board.shape[0]
        board_width = self.board.shape[1]

        horizontal_indices = self.horizontal_lines[:, 0, 0].reshape(-1)
        vertical_indices = self.vertical_lines[:, 0, 1].reshape(-1)

        horizontal_indices = np.sort(horizontal_indices)
        vertical_indices = np.sort(vertical_indices)

        horizontal_indices = Board.complete_missing_line_indices(horizontal_indices, board_width)
        vertical_indices = Board.complete_missing_line_indices(vertical_indices, board_height)

        square_horizontal_slices = [slice(horizontal_indices[i], horizontal_indices[i + 1])
                                    for i in range(len(horizontal_indices) - 1)]
        square_vertical_slices = [slice(vertical_indices[i], vertical_indices[i + 1])
                                  for i in range(len(vertical_indices) - 1)]

        square_slices = list(product(square_vertical_slices, square_horizontal_slices))
        # all_vertical_slices, all_horizontal_slices = np.array(square_slices).T
        square_indices = list(product(np.arange(len(square_vertical_slices)),
                                      np.arange(len(square_horizontal_slices))))

        board_squares_dict = dict()
        for square_idx, (row_slice, col_slice) in zip(square_indices, square_slices):
            board_squares_dict[square_idx] = self.board[row_slice, col_slice]

        return board_squares_dict

    @staticmethod
    def complete_missing_line_indices(line_indices: np.ndarray, size: int) -> np.ndarray:
        if len(line_indices) == Board.N_LINES:
            complete_line_indices = line_indices
        elif len(line_indices) == Board.N_LINES - 1:
            if line_indices[0] > size - line_indices[-1]:
                complete_line_indices = np.insert(line_indices, 0, 0)
            else:
                complete_line_indices = np.insert(line_indices, len(line_indices), size)
        else:
            complete_line_indices = np.array([0, *line_indices, size])

        return complete_line_indices

    def my_split_board_into_squares(self):
        edges = get_edge_image(self.board, lower_threshold=Canny.lower_threshold.value,
                               upper_threshold=Canny.upper_threshold.value)

        # row_seq = get_max_seq_lens_per_row(edges)
        # col_seq = get_max_seq_lens_per_row(edges.T)
        # board_squares = split_board_image_to_squares(self.board, row_seq, col_seq)

        cropped_image, cropped_edges, cropped_row_seq, cropped_col_seq = crop_image(self.board, edges, verbose=True)

        board_squares = split_board_image_to_squares(cropped_image, cropped_row_seq, cropped_col_seq)

        return board_squares



def parse_board(image: np.ndarray) -> Optional[Dict[Tuple[int, int], np.ndarray]]:
    board = detect_board(image)
    if board is None:
        logger.info("Couldn't detect a board")
        return

    board_squares = board.my_split_board_into_squares()

    return board_squares




def detect_board(image: np.ndarray) -> Optional[Board]:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    sharpened_image = cv2.filter2D(gray_image, -1, sharpen_kernel)

    threshold_image = cv2.threshold(sharpened_image, 10, 255, cv2.THRESH_BINARY_INV)[1]

    contours = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    min_area_board = 6400
    height_width_error_ratio = 0.01
    board = None
    board_count = 0
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < min_area_board:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        height_width_ratio = h / w
        if height_width_ratio <= 1 - height_width_error_ratio or height_width_ratio >= 1 + height_width_error_ratio:
            continue

        roi = image[y:y + h, x:x + w]

        board_count += 1
        candidate_board = Board(roi)
        if board is None:
            board = candidate_board
        elif candidate_board > board:
            board = candidate_board

    logger.info(f'Detected {board_count} boards. Proceeding with the largest board detected')

    return board


def detect_number_of_lines(sub_region: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    # Use Canny edge detector to find edges in the image
    gray_subregion = cv2.cvtColor(sub_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_subregion, 50, 150)

    # Find lines in the image using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=100, maxLineGap=10)

    if lines is None:
        return

    lines = np.array(lines)
    lines = lines.reshape((-1, 2, 2))
    line_points_diffs = np.abs(lines[:, 0] - lines[:, 1])
    vertical_lines = lines[line_points_diffs[:, 0] > 0, :]
    horizontal_lines = lines[line_points_diffs[:, 1] > 0, :]

    return vertical_lines, horizontal_lines

def draw_lines(image, lines, title):
    line_image = image.copy()
    for line in lines:
        cv2.line(line_image, line[0], line[1], (0, 0, 255), 2)

    # cv2.imshow(title, line_image)
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    image = cv2.imread(r"C:\Users\GoMaN\Desktop\GoMaN\Projects\BoardToFEN\dataset\full_boards\27.png")
    squares = parse_board(image)

    fig, axs = plt.subplots(8, 8)
    for (i, j), square in squares.items():
        axs[i, j].imshow(square)

    plt.show()
