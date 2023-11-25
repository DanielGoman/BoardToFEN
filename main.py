import hydra

from omegaconf import DictConfig

from src.pipeline.pipeline import Pipeline
from src.input_utils.keyboard_controler import KeyboardController
from consts import INFERENCE_CONFIG_PATH, INFERENCE_CONFIG_FILE_NAME


@hydra.main(config_path=INFERENCE_CONFIG_PATH, config_name=INFERENCE_CONFIG_FILE_NAME, version_base='1.2')
def main(config: DictConfig):
    pipeline = Pipeline(model_path=config.paths.final_model_path,
                        transforms=config.transforms)
    controller = KeyboardController(pipeline)
    controller.start_listener()


if __name__ == "__main__":
    main()
