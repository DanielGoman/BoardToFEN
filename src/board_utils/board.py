import cv2
import numpy as np
import itertools

from typing import List
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
from pprint import pprint



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


def convert_frame_to_square_vertices(row_seq: List[int], col_seq: List[int], board_size: int = 8) -> np.ndarray:
    """Uses the x and y axis' edge sequences to calculate indices for each square on the board inside the given frame

    Args:
        row_seq: list of the longest sequences of edges in each row
        col_seq: list of the longest sequences of edges in each column
        board_size: number of squares in a single row/column of the board, assumes it's the same number
                        for rows and columns

    Returns:
        squares_idx: (board_size, board_size, 4) matrix which contains (top left bottom right) indices for each
                        square of the board w.r.t the indices of the frame

    """
    x_edge_margins = find_edges(row_seq, split_size=board_size)
    y_edge_margins = find_edges(col_seq, split_size=board_size)

    # third dimension to store top left and bottom right vertices
    squares_idx = np.zeros((board_size, board_size, 4))

    # using the margins to crop the squares
    for i, (top_row_idx_margins, bottom_row_idx_margins) in enumerate(zip(x_edge_margins[:-1], x_edge_margins[1:])):
        for j, (left_col_idx_margins, right_col_idx_margins) in enumerate(zip(y_edge_margins[:-1], y_edge_margins[1:])):
            top_row_idx = top_row_idx_margins[1]
            left_row_idx = left_col_idx_margins[1]
            bottom_row_idx = bottom_row_idx_margins[0]
            right_row_idx = right_col_idx_margins[0]
            squares_idx[i, j] = (top_row_idx, left_row_idx, bottom_row_idx, right_row_idx)

    return squares_idx



def find_edges(seq: List[int], split_size, margin_size: int = 1) -> np.ndarray:
    """Finds the indices of the edges on the board along a single axis
    The axis is explicitly passed through the `seq` variable

    Args:
        seq: list of the longest sequences of edges in all rows or in all columns
        split_size: the number of squares the axis should be split to
        margin_size: margin value to add to the detected edges between squares

    Returns:
        edges_margins: list of (edge - margin, edge + margin) for all edges that separate between different squares
                        along some axis (x or y)

    """
    fragment_size = len(seq) // split_size
    split_idx = np.arange(fragment_size // 2, len(seq), fragment_size)
    fragments = np.split(seq, split_idx)

    pprint(fragments)
    print()

    padded_fragments, start_pad_size, end_pad_size = pad_fragments(fragments)

    conv_mat = np.ones((1, 3))
    conv_fragments = convolve2d(padded_fragments, conv_mat, mode='same')

    max_len_idx_per_row = np.argmax(conv_fragments, axis=1)

    # calculate absolute idx in row/column
    # start_pad_size is used to adjust the indices due to the padding
    frame_split_idx = [idx + fragment_size*i - start_pad_size for i, idx in enumerate(max_len_idx_per_row)]

    print(max_len_idx_per_row)
    print(frame_split_idx)

    # pad margins
    edges_margins = [(max(0, idx - margin_size), min(len(seq) - 1, idx + margin_size)) for idx in frame_split_idx]
    print(edges_margins)

    return edges_margins


def _print(seq1, seq2):
    for i, (val1, val2) in enumerate(zip(seq1, seq2)):
        print(f'{i}: {int(val1)}   {int(val2)}')



def pad_fragments(fragments: List[np.ndarray]) -> np.ndarray:
    padded_fragments = fragments.copy()

    required_size = len(fragments[1])

    pad_start = required_size - len(fragments[0])
    pad_end = required_size - len(fragments[-1])

    padded_fragments[0] = np.pad(padded_fragments[0], (pad_start, 0), mode='constant', constant_values=0)
    padded_fragments[-1] = np.pad(padded_fragments[-1], (0, pad_end), mode='constant', constant_values=0)

    padded_fragments = np.array(padded_fragments, dtype=int)

    return padded_fragments, pad_start, pad_end


def unpad_fragments(padded_fragments: np.ndarray, start_pad: int, end_pad: int) -> List[np.ndarray]:
    fragments = list(padded_fragments.copy())
    fragments[0] = fragments[0][start_pad:]
    fragments[-1] = fragments[-1][:-end_pad]

    return fragments


def crop_by_sum(edges):
    height, width = edges.shape[:2]
    portion = 0.6
    x_sum = np.sum(edges, axis=1)
    y_sum = np.sum(edges, axis=0)

    x_start = np.argmax(x_sum > portion * width)
    y_start = np.argmax(y_sum > portion * height)

    x_end = height - np.argmax(x_sum[::-1] > portion * width)
    y_end = width - np.argmax(y_sum[::-1] > portion * height)

    # print('Crop by sum:')
    #
    # print(f'start: ({x_start}, {y_start})')
    # print(f'end: ({x_end}, {y_end})\n')
    #
    # plt.imsave('sum_crop.png', edges[x_start: x_end, y_start: y_end], cmap='gray')


def crop_by_seq(edges: np.ndarray, portion=0.1) -> (np.ndarray, np.ndarray, np.ndarray):
    height, width = edges.shape
    row_seq = get_max_seq_lens_per_row(edges)
    col_seq = get_max_seq_lens_per_row(edges.T)

    x_start = np.argmax(row_seq > portion * width)
    y_start = np.argmax(col_seq > portion * height)

    x_end = height - np.argmax(row_seq[::-1] > portion * width)
    y_end = width - np.argmax(col_seq[::-1] > portion * height)

    cropped_row_seq = row_seq[x_start: x_end]
    cropped_col_seq = col_seq[y_start: y_end]
    cropped_edges = edges[x_start: x_end, y_start: y_end]

    return cropped_edges, cropped_row_seq, cropped_col_seq, (x_start, x_end), (y_start, y_end)


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
    _edges = cv2.Canny(image, 50, 250)
    _edges[_edges < 50] = 0
    _edges[_edges >= 50] = 1

    _cropped_edges, _cropped_row_seq, _cropped_col_seq, x_crop, y_crop = crop_by_seq(_edges)

    print('Cropped size:', _cropped_edges.shape)
    print()

    board_square_vertices = convert_frame_to_square_vertices(_cropped_edges, _cropped_row_seq, _cropped_col_seq)
    # TODO: get piece templates
    # TODO: decide how to match between templates to cropped pieces
