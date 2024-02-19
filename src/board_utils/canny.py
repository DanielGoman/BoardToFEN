from typing import Tuple

import cv2
import numpy as np

from src.board_utils.consts import Canny


def canny_edge_detector(image, low_thresh_ratio=0.05, high_thresh_ratio=0.15):
    # TODO: Understand this code and document it
    """Canny edge detection without using cv2.Canny."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_img = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edge_magnitude, edge_angle = compute_edge_gradients(blurred_img)

    # Assuming NMS and thresholding functions are implemented
    nms_result = non_max_suppression(edge_magnitude, edge_angle)  # This needs actual implementation
    high_thresh = nms_result.max() * high_thresh_ratio
    low_thresh = high_thresh * low_thresh_ratio
    thresholded = double_edge_threshold(nms_result, low_thresh, high_thresh)
    edges = weak_edge_correction(thresholded)

    edges[edges > 0] = 1

    return edges


def compute_edge_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute horizontal and vertical gradients (magnitude and angle in degrees) using Sobel operators.

    Args:
        image: image to which this method calculates the edge magnitude and angle

    Returns:
        edge_magnitude: magnitude of the edge per pixel (considering edge magnitude horizontally and vertically)
        edge_angle: angle of the edge per pixel (depends on the edge magnitude in the
                    horizontal and vertical directions)

    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = cv2.magnitude(grad_x, grad_y)
    edge_angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    return edge_magnitude, edge_angle


def non_max_suppression(edge_magnitude: np.ndarray, edge_angle: np.ndarray) -> np.ndarray:
    """Apply non-maximum suppression to thin out edges.
    For each pixel, find its neighboring pixels by its angle.
    If that pixel has larger magnitude than its neighbors then its magnitude is kept, otherwise it's zeroed out

    Args:
        edge_magnitude: magnitude of the edge per pixel
        edge_angle: angle of the edge per pixel

    Returns:
        nms_edge_image: edge image after nms

    """
    height, width = edge_magnitude.shape
    nms_edge_image = np.zeros((height, width), dtype=np.float32)
    edge_angle = edge_angle % 180  # Ensure angles are within 0-180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Determine direction of the edge
            if (0 <= edge_angle[i, j] < 22.5) or (157.5 <= edge_angle[i, j] <= 180):
                neighbors = [edge_magnitude[i, j + 1], edge_magnitude[i, j - 1]]
            elif 22.5 <= edge_angle[i, j] < 67.5:
                neighbors = [edge_magnitude[i - 1, j + 1], edge_magnitude[i + 1, j - 1]]
            elif 67.5 <= edge_angle[i, j] < 112.5:
                neighbors = [edge_magnitude[i - 1, j], edge_magnitude[i + 1, j]]
            elif 112.5 <= edge_angle[i, j] < 157.5:
                neighbors = [edge_magnitude[i - 1, j - 1], edge_magnitude[i + 1, j + 1]]
            else:
                continue

            # Suppress pixels not forming an edge
            if edge_magnitude[i, j] >= max(*neighbors):
                nms_edge_image[i, j] = edge_magnitude[i, j]
            else:
                nms_edge_image[i, j] = 0

    return nms_edge_image


def double_edge_threshold(edge_image: np.ndarray, low_thresh: float, high_thresh: float) -> np.ndarray:
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

    """
    strong_i, strong_j = np.where(edge_image >= high_thresh)
    weak_i, weak_j = np.where((edge_image <= high_thresh) & (edge_image >= low_thresh))

    thresholded = np.zeros(edge_image.shape, dtype=np.uint8)
    thresholded[strong_i, strong_j] = Canny.strong_edge.value
    thresholded[weak_i, weak_j] = Canny.weak_edge.value

    return thresholded


def weak_edge_correction(thresholded: np.ndarray) -> np.ndarray:
    """Finalize edge detection by converting weak edges connected to strong edges into strong edges.
    For any weak edge pixel, if any of the neighboring pixels of the current pixel represents a strong edge then
    the current weak edge pixel is also considered a strong edge, otherwise it's considered non-edge and is set to 0.

    Args:
        thresholded: image after double thresholding

    Returns:
        thresholded: image after deciding for each weak edge if it's not an edge or a strong edge

    """
    height, width = thresholded.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if thresholded[i, j] == Canny.weak_edge.value:
                if np.any(thresholded[i - 1:i + 2, j - 1:j + 2] == Canny.strong_edge.value):
                    thresholded[i, j] = Canny.strong_edge.value

    return thresholded
