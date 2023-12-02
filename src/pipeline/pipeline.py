import torch

from typing import List
from omegaconf import DictConfig

from src.board_utils.board import parse_board
from src.fen_converter.fen_converter import convert_pieces_to_fen
from src.input_utils.image_capture import ImageCapture
from src.model.dataset import PiecesDataset
from src.model.model import PieceClassifier
from src.utils.transforms import parse_config_transforms


class Pipeline:
    def __init__(self, model_path: str, transforms: DictConfig, model_params: dict):
        self.model = PieceClassifier(**model_params)
        self.model.load_state_dict(torch.load(model_path))
        self.transforms = parse_config_transforms(transforms)

    def run_pipeline(self) -> List[str]:
        # Capture frame of the screen
        cap = ImageCapture()
        image = cap.capture()

        # Convert the frame into a dict of squares
        board_squares = parse_board(image)

        # Predict the label of each square
        pieces_batch = PiecesDataset.board_squares_to_pieces_dataset(board_squares, self.transforms)
        predicted_labels = self.model.inference(pieces_batch)

        board_rows_as_fen = convert_pieces_to_fen(predicted_labels)

        return board_rows_as_fen






