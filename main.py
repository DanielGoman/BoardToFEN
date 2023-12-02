import hydra

from omegaconf import DictConfig

from src.gui.app import App
from src.pipeline.pipeline import Pipeline
from src.fen_converter.consts import Domains
from consts import INFERENCE_CONFIG_PATH, INFERENCE_CONFIG_FILE_NAME


@hydra.main(config_path=INFERENCE_CONFIG_PATH, config_name=INFERENCE_CONFIG_FILE_NAME, version_base='1.2')
def main(config: DictConfig):
    domain_logo_paths = {Domains[domain_name].value: domain_logo_path for domain_name, domain_logo_path in
                         config.paths.app_paths.domain_logos.items()}

    pipeline = Pipeline(model_path=config.paths.final_model_path,
                        transforms=config.transforms,
                        model_params=config.model_params)

    app = App(pipeline=pipeline,
              active_color_image_paths=config.paths.app_paths.active_color_image_paths,
              screenshot_image_path=config.paths.app_paths.screenshot_image_path,
              domain_logo_paths=domain_logo_paths)

    app.start_app()


if __name__ == "__main__":
    main()
