import torch

from typing import List, Optional
from omegaconf import DictConfig

from src.board_utils.board import parse_board
from src.fen_converter.fen_converter import convert_pieces_to_fen
from src.input_utils.image_capture import ImageCapture
from src.model.dataset import PiecesDataset
from src.model.model import PieceClassifier
from src.utils.transforms import parse_config_transforms


class Pipeline:
    """This class runs the entire sequence from screenshotting to generation of a partial FEN

    """
    def __init__(self, model_path: str, transforms: DictConfig, model_params: dict):
        """Instantiates the pre-trained model and its transforms

        Args:
            model_path: path to the weights of the trained model
            transforms: dict of transforms for the model
            model_params: parameters to the class of the pre-trained model

        """
        self.model = PieceClassifier(**model_params)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.transforms = parse_config_transforms(transforms)

    def run_pipeline(self) -> Optional[List[str]]:
        """Runs the majority of the pipeline, this includes:
            - Screenshotting
            - Conversion of the screenshot to squares of the board
            - Prediction on the board squares
            - Conversion of the predictions to a partial FEN

        Returns:
            board_rows_as_fen: list of strings - one per row, in FEN format

        """
        # Capture frame of the screen
        cap = ImageCapture()
        image = cap.capture()

        # Convert the frame into a dict of squares
        board_squares = parse_board(image)
        if board_squares is None:
            return

        # Predict the label of each square
        pieces_batch = PiecesDataset.board_squares_to_pieces_dataset(board_squares, self.transforms)
        predicted_labels = self.model.inference(pieces_batch)

        board_rows_as_fen = convert_pieces_to_fen(predicted_labels)

        return board_rows_as_fen






