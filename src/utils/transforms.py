import hydra

from omegaconf import DictConfig


def parse_config_transforms(transforms: DictConfig):
    return [hydra.utils.instantiate(transform, _convert_='partial') for transform in transforms.values()]
