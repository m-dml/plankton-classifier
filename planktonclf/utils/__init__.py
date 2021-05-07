from planktonclf.utils.Config import Config

import yaml
import os

THIS_FILE = __file__


def load_config() -> object:
    """
    Loads the config file and transforms it into a python class for easy access.
    """

    default_file = os.path.abspath(os.path.join(os.path.split(THIS_FILE)[0], "../../default_config.yaml"))
    with open(os.path.abspath(default_file), "r") as f:
        config_dict = yaml.safe_load(f)

    config_class = Config(config_dict)
    return config_class


CONFIG = load_config()
