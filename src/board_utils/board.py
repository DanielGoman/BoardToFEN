from typing import Tuple

import cv2
import numpy as np
import itertools

from matplotlib import pyplot as plt
from src.input_utils.image_capture import ImageCapture


class Board:
    def __init__(self, board_template_path):
        self.sift_feature_extractor = cv2.SIFT_create()
        self.board_frame, self.board_keypoints, self.board_descriptors = self.get_descriptors(path=board_template_path)
        self.matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    def convert_to_fen(self, frame: np.ndarray):
        _, key_points, descriptors = self.get_descriptors(frame=frame)

        matches = self.matcher.match(self.board_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        img = cv2.drawMatches(self.board_frame, self.board_keypoints, frame, key_points, matches[:50], frame, flags=2)
        cv2.imwrite('matches.png', img)

    def get_descriptors(self, frame: np.ndarray = None, path: str = None):
        if frame is None:
            if path:
                frame = cv2.imread(path)
            else:
                raise FileNotFoundError("'get_descriptor' - did not receive image nor image path")

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        key_points, descriptors = self.sift_feature_extractor.detectAndCompute(gray_frame, None)

        # img = cv2.drawKeypoints(gray_frame,
        #                         key_points,
        #                         frame,
        #                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imwrite('board_kp.png', img)

        return frame, key_points, descriptors

    @staticmethod
    def crop_by_sum(edges):
        height, width = edges.shape[:2]
        portion = 0.6
        x_sum = np.sum(edges, axis=1)
        y_sum = np.sum(edges, axis=0)

        x_start = np.argmax(x_sum > portion * width)
        y_start = np.argmax(y_sum > portion * height)

        x_end = height - np.argmax(x_sum[::-1] > portion * width)
        y_end = width - np.argmax(y_sum[::-1] > portion * height)

        print('Crop by sum:')

        print(f'start: ({x_start}, {y_start})')
        print(f'end: ({x_end}, {y_end})\n')

        plt.imsave('sum_crop.png', edges[x_start: x_end, y_start: y_end], cmap='gray')

    @staticmethod
    def crop_by_seq(edges: np.ndarray, portion=0.1) -> (np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]):
        height, width = edges.shape
        row_seq = Board.get_max_seq_lens_per_row(edges)
        col_seq = Board.get_max_seq_lens_per_row(edges.T)

        x_start = np.argmax(row_seq > portion * width)
        y_start = np.argmax(col_seq > portion * height)

        x_end = height - np.argmax(row_seq[::-1] > portion * width)
        y_end = width - np.argmax(col_seq[::-1] > portion * height)

        print('Crop by seq:')

        print(f'start: ({x_start}, {y_start})')
        print(f'end: ({x_end}, {y_end})\n')

        plt.imsave('seq_crop.png', edges[x_start: x_end, y_start: y_end], cmap='gray')

        return row_seq, col_seq, (x_start, x_end), (y_start, y_end)

    @staticmethod
    def get_max_seq_lens_per_row(frame: np.ndarray) -> np.ndarray:
        """Retrieves the length of the longest edge in all rows of the given matrix

        Args:
            frame: the matrix in which we search for the longest edge in each row

        Returns:
            max_seq_len_per_row_no_holes: index i contains the length of the longest edge in row i

        """
        height, width = frame.shape[:2]

        # padding each row to also capture sequences that start from the first index
        padding = np.zeros((height, 1))
        stacked = np.hstack([padding, frame == 1, padding])

        # subtracting consecutive indexes
        diff = np.diff(stacked)

        # Finds the start and end of all the sequences
        rows_number, seq_idx = np.where(diff)

        # number of the sequences found per row - if found 2 sequences in some row, then the lengthfor that row
        # would be 4:   start_idx_seq1, end_idx_seq1, start_idx_seq2, end_idx_seq2
        row_unique_seqs_number = [len(list(values)) for key, values in itertools.groupby(rows_number)]
        # start and end of the sequences list of each row
        row_seq_idx_split = np.cumsum(row_unique_seqs_number)

        # separate row-wise the sets of sequence indexes and calculate the length of each sequence
        seq_idx_list_per_row = np.split(seq_idx, row_seq_idx_split)[:-1]
        seq_idx_list_per_row = [row_seq_indices.reshape(-1, 2) for row_seq_indices in seq_idx_list_per_row]
        seq_lens_per_row = [np.diff(row_seq_indices) for row_seq_indices in seq_idx_list_per_row]

        # calculate the maximal length of a sequence in each row
        max_seq_len_per_row = [np.max(row_seq_lens) for row_seq_lens in seq_lens_per_row]

        # single list with max length of the sequence in each row.
        # rows with no sequence at all get a 0 value
        max_seq_len_per_row_no_holes = np.zeros(height)
        existent_rows = np.unique(rows_number)
        np.put(max_seq_len_per_row_no_holes, existent_rows, max_seq_len_per_row)

        return max_seq_len_per_row_no_holes


if __name__ == "__main__":
    path = "../../data/real_board.png"
    image = cv2.imread(path)

    print('Image shape:', image.shape[:2])
    print()

    # edge detector
    edges = cv2.Canny(image, 50, 250)
    edges[edges < 50] = 0
    edges[edges >= 50] = 1

    row_seq, col_seq, x_crop_idx, y_crop_idx = Board.crop_by_seq(edges)

