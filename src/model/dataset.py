import json
import os
import yaml

import hydra
import torch
import torch.nn.functional as F
import torchvision

from omegaconf import DictConfig
from PIL import Image

from torch.utils.data import Dataset
from typing import Dict, List

from src.data.consts.piece_consts import PIECE_TYPE, PIECE_COLOR, NON_PIECE


class PiecesDataset(Dataset):
    def __init__(self, images_dir_path: str, labels_path, transforms: list = None, dtype='float32',
                 images_extension: str = 'png', device: str = 'cpu', minibatch_size: int = None):
        self.dtype = dtype
        self.images_extension = images_extension
        self.device = device
        self.minibatch_size = minibatch_size

        if transforms:
            self.transforms = torchvision.transforms.Compose(transforms)

        self.labels_dict = self.load_labels(labels_path)

        num_images = len(os.listdir(images_dir_path))
        num_labels = len(self.labels_dict)
        if minibatch_size == -1 and num_images != num_labels:
            raise FileExistsError(f"Number of images ({num_images}) does not match number of labels ({num_labels})")

        self.image_path_labels_pairs = self.create_imagepath_labels_pairs(images_dir_path, self.labels_dict)

    def load_labels(self, labels_path: str) -> Dict[str, Dict[str, str]]:
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
                        square_labels = {key: item for key, item in items.items() if key != 'image_file_name'}
                        single_labels_dict[square_name] = square_labels

                        num_collected_squares += 1
                        if num_collected_squares == self.minibatch_size:
                            return dict(sorted(single_labels_dict.items()))

            single_labels_dict = dict(sorted(single_labels_dict.items()))

        return single_labels_dict

    def create_imagepath_labels_pairs(self, images_dir_path: str, labels_dict: Dict[str, Dict[str, str]]) \
            -> List[Dict[str, Dict[str, torch.Tensor]]]:
        """Creates a list that for every image name contains path to the respective image and the labels

        Args:
            images_dir_path: path to the directory that contains the images
            labels_dict: the dictionary that contains the labels for each image

        Returns:
            path_labels_pairs_dict: list of pairs of (image_path, image_labels) for every image name

        """
        one_hot_piece_type, one_hot_piece_color = self.transform_labels(labels_dict)
        is_piece = torch.Tensor([int(piece_info['piece_type'] != NON_PIECE) for piece_info in labels_dict.values()])
        path_labels_pairs_dict = []
        for idx, image_name in enumerate(labels_dict):
            full_image_name = f'{image_name}.{self.images_extension}'
            image_path = os.path.join(images_dir_path, full_image_name)
            image = Image.open(image_path)

            one_hot_image_labels = {'piece_type': one_hot_piece_type[idx],
                                    'piece_color': one_hot_piece_color[idx],
                                    'is_piece': is_piece[idx]}

            path_labels_pair = {'image': image,
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

        one_hot_types = F.one_hot(torch.Tensor(piece_type_labels).to(torch.long),
                                  num_classes=len(PIECE_TYPE)).to(torch.float64)
        one_hot_colors = F.one_hot(torch.Tensor(piece_color_labels).to(torch.long),
                                   num_classes=len(PIECE_COLOR)).to(torch.float64)

        one_hot_colors = one_hot_colors[:, :-1]

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
        image = imagepath_labels_pair['image']
        image_labels = imagepath_labels_pair['labels']

        piece_type = image_labels['piece_type']
        piece_color = image_labels['piece_color']
        is_piece = image_labels['is_piece']

        transformed_image = self.transforms(image)

        return transformed_image, piece_type, piece_color, is_piece


@hydra.main(config_path=r'../../configs/', config_name=r'train.yaml', version_base='1.2')
def test_run(config: DictConfig):
    images_dir_path_ = config.paths.data_paths.image_dir_path
    labels_path_ = config.paths.data_paths.labels_json_path
    dataset = PiecesDataset(images_dir_path=images_dir_path_,
                            labels_path=labels_path_)


if __name__ == "__main__":
    test_run()

