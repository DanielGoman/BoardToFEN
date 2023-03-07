import json
import os
import torch
import torchvision

from torch.utils.data import Dataset
from typing import Dict

from src.data.consts.piece_consts import PIECE_TYPE, PIECE_COLOR


class PiecesDataset(Dataset):
    def __init__(self, images_dir_path: str, labels_path, transforms, dtype='float32', images_extension: str = 'png'):
        self.dtype = dtype
        self.transforms = transforms
        self.images_extension = images_extension
        self.labels_dict = self.load_labels(labels_path)
        # self.labels = self.transform_labels(self.labels_dict)

        num_images = len(os.listdir(images_dir_path))
        num_labels = len(self.labels_dict)
        if num_images != num_labels:
            raise FileExistsError(f"Number of images ({num_images}) does not match number of labels ({num_labels})")

        self.image_path_labels_pairs = self.create_imagepath_labels_pairs(images_dir_path, self.labels_dict)
        a = 5

    @staticmethod
    def load_labels(labels_path: str) -> Dict[str, Dict[str, str]]:
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

    def create_imagepath_labels_pairs(self, images_dir_path: str, labels_dict: Dict[str, Dict[str, str]]) \
                                        -> Dict[str, Dict[str, Dict[str, str]]]:
        path_labels_pairs_dict = {}
        for image_name, image_labels in labels_dict.items():
            full_image_name = f'{image_name}.{self.images_extension}'
            image_path = os.path.join(images_dir_path, full_image_name)
            path_labels_pair = {'image_path': image_path,
                                'labels': image_labels}
            path_labels_pairs_dict[image_name] = path_labels_pair

        return path_labels_pairs_dict

    def transform_labels(self, labels_dict: Dict[str, Dict[str, str]]):
        piece_type_labels = [PIECE_TYPE[item['piece_type']] for item in labels_dict.values()]
        piece_color_labels = [PIECE_COLOR[item['piece_color']] for item in labels_dict.values()]

        one_hot_types = keras.utils.to_categorical(piece_type_labels, num_classes=len(PIECE_TYPE), dtype=self.dtype)
        one_hot_colors = keras.utils.to_categorical(piece_color_labels, num_classes=len(PIECE_COLOR), dtype=self.dtype)

        # TODO: complete the transform into a single matrix with type and color feature encoded
        #       proceed to loading the image dataset and applying transformations

    def __len__(self):
        pass

    def __getitem__(self, idx: int):
        pass


if __name__ == "__main__":
    images_dir_path_ = r'../../dataset/squares'
    labels_path_ = r'../../dataset/labels/labels.json'
    PiecesDataset(images_dir_path=images_dir_path_,
                  labels_path=labels_path_,
                  transforms=None)
