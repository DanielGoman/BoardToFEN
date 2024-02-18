import cv2
import itertools
import numpy as np

from typing import List, Dict, Tuple
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

from src.board_utils.consts import Canny
from src.data.consts.squares_consts import BOARD_SIDE_SIZE



def parse_board(image: np.ndarray, verbose: bool = False) -> Dict[Tuple[int, int], np.ndarray]:
    """Parses an image that presumably contains a chess board, returning a dictionary of squares of the board

    Args:
        image: an image that contains a chess board. The image can also include background to an extent
        verbose: prints information during runtime if True, otherwise False

    Returns:
        board_squares: "2d" dict of squares.
                        board_squares[(square_x, square_y)] = np.ndarray of the square

    """
    edges = get_edge_image(image, lower_threshold=Canny.lower_threshold.value,
                           upper_threshold=Canny.upper_threshold.value)

    cropped_image, cropped_edges, cropped_row_seq, cropped_col_seq = crop_image(image, edges, verbose=verbose)

    board_squares = split_board_image_to_squares(cropped_image, cropped_row_seq, cropped_col_seq)

    return board_squares


def crop_image(image: np.ndarray, edges: np.ndarray, verbose: bool) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crops an image to a smaller image that contains only the chess board

    Args:
        image: input image
        edges: binary edge mask of the input image
        verbose: prints information during runtime if True, otherwise False

    Returns:
        cropped_image: the input image, cropped to contain only the chess board inside it
        cropped_edges: the binary edge mask, cropped to contain only the chess board inside it
        cropped_row_seq: the longest row sequence in the edge mask image, according to which the width of the cropped
                            image was selected
        cropped_col_seq: the longest col sequence in the edge mask image, according to which the height of the cropped
                            image was selected

    """
    cropped_edges, cropped_row_seq, cropped_col_seq, x_crop, y_crop = crop_by_seq(edges)
    cropped_image = image[x_crop[0]: x_crop[1], y_crop[0]: y_crop[1]]

    if verbose:
        print('Original size:', image.shape)
        print('Cropped size:', cropped_image.shape)

    return cropped_image, cropped_edges, cropped_row_seq, cropped_col_seq


def split_board_image_to_squares(image: np.ndarray, cropped_row_seq: np.ndarray, cropped_col_seq: np.ndarray,
                                 board_side_size: int = BOARD_SIDE_SIZE) -> Dict[Tuple[int, int], np.ndarray]:
    """Converts an image of a board to a dict of squares

    Args:
        image: a cropped image of a board
        cropped_row_seq: the longest row sequence in the edge mask image, according to which the width of the cropped
                            image was selected
        cropped_col_seq: the longest col sequence in the edge mask image, according to which the height of the cropped
                            image was selected
        board_side_size: the sizes of the board (assumes the board is of equal height and width)

    Returns:
        board_squares: "2d" dict of squares.
                        board_squares[(square_x, square_y)] = np.ndarray of the square

    """
    board_square_vertices = convert_frame_to_square_vertices(cropped_row_seq, cropped_col_seq)

    board_squares = {}

    fig, axs = plt.subplots(board_side_size, board_side_size)
    for i in range(board_side_size):
        for j in range(board_side_size):
            square_idx = board_square_vertices[i, j]
            square = image[square_idx[0]: square_idx[2], square_idx[1]: square_idx[3]]
            board_squares[(i, j)] = square
            axs[i][j].imshow(square)
    plt.show()

    return board_squares


def convert_frame_to_square_vertices(row_seq: np.ndarray, col_seq: np.ndarray,
                                     board_size: int = BOARD_SIDE_SIZE) -> np.ndarray:
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

    return squares_idx.astype(int)


def find_edges(seq: np.ndarray, split_size, margin_size: int = 1) -> List[Tuple[int, int]]:
    """Finds the indices of the edges on the board along a single axis
    The axis is explicitly passed through the `seq` variable
    - In order to do this, the board is split into `split_size` + 1 fragments, to ensure each fragment contains exactly
        one major edge of the board.
        For example, a board of width 8 will have 7 major inner edges and 2 major outer edges, 9 in total.
    - We apply a filter to find the exact index of the edge in each fragment, so that the splitting into squares
        will be as accurate as possible

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

    padded_fragments, start_pad_size, end_pad_size = pad_fragments(fragments)

    # convolving the sequence with a filter to account for edges spreading over more than a single pixel
    conv_filter = [0.5, 1, 0.5]
    conv_filter = np.array(conv_filter).reshape((-1, 1))
    conv_fragments = convolve2d(padded_fragments, conv_filter, mode='same')

    max_len_idx_per_row = np.argmax(conv_fragments, axis=1)

    # calculate absolute idx in row/column
    # start_pad_size is used to adjust the indices due to the padding
    frame_split_idx = [idx + fragment_size * i - start_pad_size for i, idx in enumerate(max_len_idx_per_row)]

    # pad margins
    edges_margins = [(max(0, idx - margin_size), min(len(seq) - 1, idx + margin_size)) for idx in frame_split_idx]

    return edges_margins


def pad_fragments(fragments: List[np.ndarray]) -> Tuple[np.ndarray, int, int]:
    """Pads the list of numpy array so that each array is of the same shape
    Essentially only the first and last fragment should be needing a padding, as it is expected from a digital board
    to have all of its squares of exactly the same shape.
    The sizes of the edges of the board may vary, and for this reason we pad them.
    This is done only to allow using matrices for the convolution step.

    Args:
        fragments: partitions of the edge sequence, such that each fragment includes exactly one major edge

    Returns:
        padded_fragments: single np.ndarray after applying padding to the first and last fragment
        pad_start: the number of pixels that were used to pad the first fragment
        pad_end: the number of pixels that were used to pad the last fragment

    """
    padded_fragments = fragments.copy()

    required_size = len(fragments[1])

    pad_start = required_size - len(fragments[0])
    pad_end = required_size - len(fragments[-1])

    padded_fragments[0] = np.pad(padded_fragments[0], (pad_start, 0), mode='constant', constant_values=0)
    padded_fragments[-1] = np.pad(padded_fragments[-1], (0, pad_end), mode='constant', constant_values=0)

    padded_fragments = np.array(padded_fragments, dtype=int)

    return padded_fragments, pad_start, pad_end


def crop_by_seq(edges: np.ndarray, portion=0.1) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Crops the binary edges mask to contain only the board

    Args:
        edges: uncropped binary edges mask
        portion: the minimal proportion of the image that an edge need to take up to be identified as an edge of the
                    board. This can be problematic if a very large image is selected, which contains long edges
                    beside the board.

    Returns:
        cropped_edges: the binary edge mask, cropped to contain only the chess board inside it
        cropped_row_seq: the longest row sequence in the edge mask image, according to which the width of the cropped
                            image was selected
        cropped_col_seq: the longest col sequence in the edge mask image, according to which the height of the cropped
                            image was selected
        x_crop_indices: the indices of the start and end of the horizontal crop, any index outside this range is cropped
        y_crop_indices: the indices of the start and end of the vertical crop, any index outside this range is cropped

    """
    height, width = edges.shape
    row_seq = get_max_seq_lens_per_row(edges)
    col_seq = get_max_seq_lens_per_row(edges.T)

    x_start = int(np.argmax(row_seq > portion * width))
    y_start = int(np.argmax(col_seq > portion * height))

    x_end = height - np.argmax(row_seq[::-1] > portion * width)
    y_end = width - np.argmax(col_seq[::-1] > portion * height)

    cropped_row_seq = row_seq[x_start: x_end]
    cropped_col_seq = col_seq[y_start: y_end]
    cropped_edges = edges[x_start: x_end, y_start: y_end]
    x_crop_indices = (x_start, x_end)
    y_crop_indices = (y_start, y_end)

    return cropped_edges, cropped_row_seq, cropped_col_seq, x_crop_indices, y_crop_indices


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

    # number of the sequences found per row - if found 2 sequences in some row, then the length for that row
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
    path = "../../dataset/full_boards/31.png"
    _image = cv2.imread(path)

    parse_board(_image, verbose=True)
