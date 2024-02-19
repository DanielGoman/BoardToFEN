from typing import Tuple

import cv2
import numpy as np

from src.board_utils.consts import Canny


def canny_edge_detector(image, low_thresh_ratio=0.05, high_thresh_ratio=0.15):
    # TODO: Understand this code and document it
    """Canny edge detection without using cv2.Canny."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_img = cv2.GaussianBlur(gray, (5, 5), 1.4)
    magnitude, angle = compute_gradients(blurred_img)

    # Assuming NMS and thresholding functions are implemented
    nms_result = non_max_suppression(magnitude, angle)  # This needs actual implementation
    high_thresh = nms_result.max() * high_thresh_ratio
    low_thresh = high_thresh * low_thresh_ratio
    thresholded, weak_edge, strong_edge = double_edge_threshold(nms_result, low_thresh, high_thresh)
    edges = edge_tracking(thresholded, weak_edge, strong_edge)

    edges[edges > 0] = 1

    return edges


def compute_gradients(image):
    """Compute horizontal and vertical gradients using Sobel operators."""
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)
    return magnitude, angle


def non_max_suppression(magnitude, angle):
    """Apply non-maximum suppression to thin out edges."""
    M, N = magnitude.shape
    Z = np.zeros((M,N), dtype=np.float32)
    angle = angle % 180  # Ensure angles are within 0-180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                # Determine directions
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    neighbors = [magnitude[i, j+1], magnitude[i, j-1]]
                elif (22.5 <= angle[i, j] < 67.5):
                    neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                elif (67.5 <= angle[i, j] < 112.5):
                    neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                elif (112.5 <= angle[i, j] < 157.5):
                    neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]

                # Suppress pixels not forming an edge
                if magnitude[i, j] >= max(neighbors):
                    Z[i, j] = magnitude[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def double_edge_threshold(edge_image: np.ndarray, low_thresh: float, high_thresh: float) -> Tuple[np.ndarray, int, int]:
    """Apply double thresholding to distinguish strong, weak, and non-edges.
    edge_image[i, j] is a strong edge if edge_image[i, j] > high_threshold (and is given `strong edge` value)
    edge_image[i, j] is a weak edge if low_thresh < edge_image[i, j] < high_thresh (and is given `weak edge` value)
    otherwise edge_image[i, j] isn't considered an edge at all (and is given a value of zero after thresholding)

    Args:
        edge_image: edge image after edge detection using Sobel filter
        low_thresh: low threshold
        high_thresh: high threshold

    Returns:
        thresholded: thresholded edge image
        weak_edge: value given to weak edges
        strong_edge: value given to strong edges

    """
    strong_edge = Canny.strong_edge.value
    weak_edge = Canny.weak_edge.value

    strong_i, strong_j = np.where(edge_image >= high_thresh)
    weak_i, weak_j = np.where((edge_image <= high_thresh) & (edge_image >= low_thresh))

    thresholded = np.zeros(edge_image.shape, dtype=np.uint8)
    thresholded[strong_i, strong_j] = strong_edge
    thresholded[weak_i, weak_j] = weak_edge

    return thresholded, weak_edge, strong_edge


def edge_tracking(thresholded, weak, strong=255):
    """Finalize edge detection by converting weak edges connected to strong edges into strong edges."""
    M, N = thresholded.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if thresholded[i, j] == weak:
                if np.any(thresholded[i - 1:i + 2, j - 1:j + 2] == strong):
                    thresholded[i, j] = strong
                else:
                    thresholded[i, j] = 0
    return thresholded
