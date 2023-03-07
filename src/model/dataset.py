import json
import os

import torch
import torch.nn.functional as F
import torchvision

from PIL import Image

from torch.utils.data import Dataset
from typing import Dict, List

from src.data.consts.piece_consts import PIECE_TYPE, PIECE_COLOR
from src.model.consts import default_transforms


class PiecesDataset(Dataset):
    def __init__(self, images_dir_path: str, labels_path, transforms: list = default_transforms, dtype='float32', images_extension: str = 'png',
                 device: str = 'cpu'):
        self.dtype = dtype
        self.images_extension = images_extension
        self.device = device
        self.transforms = torchvision.transforms.Compose(transforms)
        self.labels_dict = self.load_labels(labels_path)
        # self.labels = self.transform_labels(self.labels_dict)

        num_images = len(os.listdir(images_dir_path))
        num_labels = len(self.labels_dict)
        if num_images != num_labels:
            raise FileExistsError(f"Number of images ({num_images}) does not match number of labels ({num_labels})")

        self.image_path_labels_pairs = self.create_imagepath_labels_pairs(images_dir_path, self.labels_dict)

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

            for board_type_dict in labels_dict.values():
                for board in board_type_dict.values():
                    board_pieces = {'.'.join(items['image_file_name'].split('.')[:-1]):
                                        {key: item for key, item in items.items() if key != 'image_file_name'}
                                    for items in board.values()}

                    single_labels_dict.update(board_pieces)

            single_labels_dict = dict(sorted(single_labels_dict.items()))

        return single_labels_dict

    def create_imagepath_labels_pairs(self, images_dir_path: str, labels_dict: Dict[str, Dict[str, str]]) \
            -> List[Dict[str, Dict[str, str]]]:
        """Creates a list that for every image name contains path to the respective image and the labels

        Args:
            images_dir_path: path to the directory that contains the images
            labels_dict: the dictionary that contains the labels for each image

        Returns:
            path_labels_pairs_dict: list of pairs of (image_path, image_labels) for every image name

        """
        one_hot_piece_type, one_hot_piece_color = self.transform_labels(labels_dict)
        path_labels_pairs_dict = []
        for idx, image_name in enumerate(labels_dict):
            full_image_name = f'{image_name}.{self.images_extension}'
            image_path = os.path.join(images_dir_path, full_image_name)

            one_hot_image_labels = {'piece_type': one_hot_piece_type[idx],
                                    'piece_color': one_hot_piece_color[idx]}

            path_labels_pair = {'image_path': image_path,
                                'labels': one_hot_image_labels}
            path_labels_pairs_dict.append(path_labels_pair)

        return path_labels_pairs_dict

    @staticmethod
    def transform_labels(labels_dict: Dict[str, Dict[str, str]]) -> (torch.Tensor, torch.Tensor):
        """Transforms the labels into one hot encodings

        Args:
            labels_dict: the dictionary that contains the labels for each image

        Returns:
            one_hot_types: a tensor of one hot encodings of the piece types
            one_hot_colors: a tensor of one hot encodings of the piece colors

        """
        piece_type_labels = [PIECE_TYPE[item['piece_type']] for item in labels_dict.values()]
        piece_color_labels = [PIECE_COLOR[item['piece_color']] for item in labels_dict.values()]

        one_hot_types = F.one_hot(torch.Tensor(piece_type_labels).to(torch.long))
        one_hot_colors = F.one_hot(torch.Tensor(piece_color_labels).to(torch.long))

        return one_hot_types, one_hot_colors

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
        imagepath_labels_pair = self.image_path_labels_pairs[idx]
        image_path = imagepath_labels_pair['image_path']
        image_labels = imagepath_labels_pair['labels']

        piece_type = image_labels['piece_type']
        piece_color = image_labels['piece_color']

        image = Image.open(image_path)
        transformed_image = self.transforms(image)

        return transformed_image, piece_type, piece_color


if __name__ == "__main__":
    images_dir_path_ = r'../../dataset/squares'
    labels_path_ = r'../../dataset/labels/labels.json'
    dataset = PiecesDataset(images_dir_path=images_dir_path_,
                            labels_path=labels_path_)


