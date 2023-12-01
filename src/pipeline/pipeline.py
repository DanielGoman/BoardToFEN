import torch

from omegaconf import DictConfig

from src.board_utils.board import parse_board
from src.input_utils.image_capture import ImageCapture
from src.model.dataset import PiecesDataset
from src.utils.transforms import parse_config_transforms


class Pipeline:
    def __init__(self, model_path: str, transforms: DictConfig):
        self.model = torch.jit.load(model_path)
        self.transforms = parse_config_transforms(transforms)

    def run_pipeline(self):
        # Capture frame of the screen
        cap = ImageCapture()
        image = cap.capture()

        # Convert the frame into a dict of squares
        board_squares = parse_board(image)

        # Predict the label of each square
        pieces_batch = PiecesDataset.board_squares_to_pieces_dataset(board_squares, self.transforms)
        predicted_labels = self.model.inference(pieces_batch)

        return predicted_labels






