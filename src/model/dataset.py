import json
import torch
import torchvision

from typing import Dict

from src.data.consts.piece_consts import PIECE_TYPE, PIECE_COLOR


class PiecesDataset:
    def __init__(self, images_dir_path: str, labels_path, transforms, dtype = 'float32'):
        self.dtype = dtype
        self.labels_dict = self.load_labels(labels_path)
        self.labels = self.transform_labels(self.labels_dict)

    @staticmethod
    def load_labels(labels_path: str) -> torch.Tensor:
        with open(labels_path, 'r') as labels_json:
            labels_dict = json.load(labels_json)
            single_labels_dict = {}

            for board_type_dict in labels_dict.values():
                for board in board_type_dict.values():
                    board_pieces = {'.'.join(items['image_file_name'].split('.')[:-1]):
                                        {key: item for key, item in items.items() if key != 'image_file_name'}
                                    for items in board.values()}

                    single_labels_dict.update(board_pieces)

            single_labels_dict = dict(sorted(single_labels_dict.items()))

        return single_labels_dict

    def transform_labels(self, labels_dict: Dict[str, Dict[str, str]]):
        piece_type_labels = [PIECE_TYPE[item['piece_type']] for item in labels_dict.values()]
        piece_color_labels = [PIECE_COLOR[item['piece_color']] for item in labels_dict.values()]

        one_hot_types = keras.utils.to_categorical(piece_type_labels, num_classes=len(PIECE_TYPE), dtype=self.dtype)
        one_hot_colors = keras.utils.to_categorical(piece_color_labels, num_classes=len(PIECE_COLOR), dtype=self.dtype)

        # TODO: complete the transform into a single matrix with type and color feature encoded
        #       proceed to loading the image dataset and applying transformations


if __name__ == "__main__":
    PiecesDataset(images_dir_path=None,
                  labels_path='../../dataset/labels/labels.json',
                  transforms=None)
