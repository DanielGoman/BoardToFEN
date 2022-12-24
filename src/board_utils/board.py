import cv2
import numpy as np

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

    def crop_by_sum(self, edges):
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


if __name__ == "__main__":
    board_template_path = r'../../data/board_template.png'
    board = Board(board_template_path=board_template_path)

    cap = ImageCapture()
    img = cap.capture()
    board.get_descriptors(frame=img)

