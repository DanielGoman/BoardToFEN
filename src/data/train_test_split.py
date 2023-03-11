import json
from typing import List, Tuple, Dict

import hydra
import numpy as np

from omegaconf import DictConfig

from src.consts import TRAIN_CONFIG_PATH, TRAIN_CONFIG_NAME
from src.data.consts.path_consts import LABELS_OUTPUT_FILE_PATH, TRAIN_LABELS_OUTPUT_FILE_PATH, \
    VAL_LABELS_OUTPUT_FILE_PATH


@hydra.main(config_path=TRAIN_CONFIG_PATH, config_name=TRAIN_CONFIG_NAME, version_base='1.2')
def train_test_split(config: DictConfig):
    """Split the entire dataset into a train set and validation set according to desired train size (config param)

    Args:
        config: hydra config manager

    """
    train_size_ratio = config.hyperparams.train.train_size
    with open(LABELS_OUTPUT_FILE_PATH) as file:
        labels = json.load(file)
        single_labels_list = []
        for board_type_name, dict_of_squares_for_board_type in labels.items():
            for board_name, board_item in dict_of_squares_for_board_type.items():
                for square_item in board_item.items():
                    single_labels_list.append((board_type_name, (board_name, square_item)))

        num_samples = len(single_labels_list)

        train_size = int(num_samples * train_size_ratio)
        train_samples_idx = np.random.choice(num_samples, train_size, replace=False)
        val_samples_idx = np.array(list(set(np.arange(num_samples)) - set(train_samples_idx)))

        create_subset_from_selected_indices(single_labels_list, selected_idx=train_samples_idx,
                                            board_types_names=labels.keys(),
                                            out_json_path=TRAIN_LABELS_OUTPUT_FILE_PATH)
        create_subset_from_selected_indices(single_labels_list, selected_idx=val_samples_idx,
                                            board_types_names=labels.keys(),
                                            out_json_path=VAL_LABELS_OUTPUT_FILE_PATH)


# TODO: load train and val separately
def create_subset_from_selected_indices(dataset: List[Tuple[str, dict]], selected_idx: np.ndarray,
                                        board_types_names: List[str], out_json_path: str):
    """Creates a subset of the given dataset and saves it into a json file
    structure of subset (going from outer-most to the inner-most):
        1 - dict that contains a dict per board type (standard board, or just kings and queens)
        2 - dict of all boards of a specific type
        3 - dict of all pieces of a specific board
        4 - dict with all the information of a specific square

    Args:
        dataset: the entire dataset, laid out as a list of squares (of all board of all board types)
        selected_idx: the indices in the dataset selected to be in the subset
        board_types_names: names fo all board types in the data
        out_json_path: path to the output json which will contain the new subset

    """
    subset = {board_type_name: {} for board_type_name in board_types_names}
    for idx in selected_idx:
        board_type_name, (board_name, (key, value)) = dataset[idx]
        if board_name not in subset[board_type_name]:
            subset[board_type_name][board_name] = {}
        subset[board_type_name][board_name][key] = value

    with open(out_json_path, 'w') as out_json:
        json.dump(subset, out_json, indent=4)


if __name__ == "__main__":
    train_test_split()
