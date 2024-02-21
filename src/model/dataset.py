import json
import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from omegaconf import DictConfig
from PIL import Image

from torch.utils.data import Dataset
from typing import Dict, List, Tuple

from src.data.consts.piece_consts import LABELS


class PiecesDataset(Dataset):
    def __init__(self, images_dir_path: str, labels_path, transforms: list = None, dtype='float32',
                 images_extension: str = 'png', device: str = 'cpu'):
        self.dtype = dtype
        self.images_extension = images_extension
        self.device = device

        if transforms:
            self.transforms = torchvision.transforms.Compose(transforms)

        self.labels_dict = self.load_labels(labels_path)

        self.image_path_labels_pairs = self.create_image_labels_pairs(images_dir_path, self.labels_dict)

    @staticmethod
    def load_labels(labels_path: str) -> Dict[str, Dict[str, str]]:
        """Loads all the labels from a json file into a dictionary

        Args:
            labels_path: path to the json file that contains all the labels

        Returns:
            single_labels_dict: dictionary that for every image name contains respective labels of
                                    'piece_color' and 'piece_type'

        """
        with open(labels_path, 'r') as labels_json:
            labels_dict = json.load(labels_json)
            single_labels_dict = {}

            num_collected_squares = 0
            for board_type_dict in labels_dict.values():
                for board in board_type_dict.values():
                    for i, items in enumerate(board.values()):
                        square_name = '.'.join(items['image_file_name'].split('.')[:-1])
                        label = items['label']
                        single_labels_dict[square_name] = label

                        num_collected_squares += 1

            single_labels_dict = dict(sorted(single_labels_dict.items()))

        return single_labels_dict

    def create_image_labels_pairs(self, images_dir_path: str, labels_dict: Dict[str, Dict[str, str]]) \
            -> List[Dict[str, Dict[str, torch.Tensor]]]:
        """Creates a list that for every image name contains path to the respective image and the labels

        Args:
            images_dir_path: path to the directory that contains the images
            labels_dict: the dictionary that contains the labels for each image

        Returns:
            path_labels_pairs_dict: list of pairs of (image_path, image_labels) for every image name

        """
        one_hot_labels = self.transform_labels(labels_dict)
        image_label_pairs_dict = []
        for idx, image_name in enumerate(labels_dict):
            full_image_name = f'{image_name}.{self.images_extension}'
            image_path = os.path.join(images_dir_path, full_image_name)
            image = Image.open(image_path)

            path_label_pair = {'image': image,
                                'label': one_hot_labels[idx]}
            image_label_pairs_dict.append(path_label_pair)

        return image_label_pairs_dict

    @staticmethod
    def transform_labels(labels_dict: Dict[str, Dict[str, str]]) -> (torch.Tensor, torch.Tensor):
        """Transforms the labels into one hot encodings

        Args:
            labels_dict: the dictionary that contains the labels for each image

        Returns:
            one_hot_types: a tensor of one hot encodings of the piece types
            one_hot_colors: a tensor of one hot encodings of the piece colors

        """
        labels = [LABELS[square_class] for square_class in labels_dict.values()]

        one_hot_label = F.one_hot(torch.Tensor(labels).to(torch.long),
                                  num_classes=len(LABELS)).to(torch.float64)

        return one_hot_label

    @staticmethod
    def board_squares_to_pieces_dataset(board_squares: Dict[Tuple[int, int], np.ndarray],
                                        transforms: list = None) -> torch.Tensor:
        """Transforms the 2d list of squares into PiecesDataset

        Args:
            board_squares: "2d" dict of squares.
                            board_squares[(square_x, square_y)] = np.ndarray of the square
            transforms: list of transforms to be applied to the squares

        Returns:
            - PiecesDataset that contains all the squares from `board_squares`
            - number of squares on the board

        """
        squares = list(board_squares.values())

        if transforms:
            squares_pil_images = [Image.fromarray(square_array.astype(np.uint8)) for square_array in squares]
            transforms = torchvision.transforms.Compose(transforms)
            squares = [transforms(image) for image in squares_pil_images]

        squares_tensor = torch.stack(squares)
        return squares_tensor

    def __len__(self):
        return len(self.image_path_labels_pairs)

    def __getitem__(self, idx: int):
        """getitem method for this class

        Args:
            idx: index of the element to retrieve

        Returns:
            transformed_image: input image after transformations
            piece_type: one hot encoding of piece type
            piece_color: one hot encoding of the piece color

        """
        image_label_pair = self.image_path_labels_pairs[idx]
        image = image_label_pair['image']
        image_label = image_label_pair['label']

        transformed_image = self.transforms(image)

        return transformed_image, image_label


@hydra.main(config_path=r'../../configs/', config_name=r'train.yaml', version_base='1.2')
def test_run(config: DictConfig):
    images_dir_path_ = config.paths.data_paths.image_dir_path
    labels_path_ = config.paths.data_paths.labels_json_path
    dataset = PiecesDataset(images_dir_path=images_dir_path_,
                            labels_path=labels_path_)


if __name__ == "__main__":
    test_run()

