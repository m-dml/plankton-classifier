import os
from argparse import ArgumentParser

import yaml

from src.utils import CONFIG


def load_config():
    parser = ArgumentParser()
    parser.add_argument("--config_file", "-f", type=str, default="default_config.yaml",
                        help="Set the configuration file used for the experiment.")

    args = parser.parse_args()
    with open(os.path.abspath(args.config_file), "r") as f:
        config_dict = yaml.safe_load(f)

    # update values in the config class.
    CONFIG.update(config_dict)


if __name__ == '__main__':
    load_config()
