import sys, os
sys.path.append(os.path.abspath("../"))

from src.models.catboost_meta_classifier import BoostClassifier
from src.utils import CONFIG
from src.utils.DataLoader import PlanktonDataLoader


if __name__ == "__main__":
    ONNX_FILE = ""  # is not used so it can be an empty string
    CONFIG_FILE = "../strand_config.yaml"
    CPUS = -1
    model = BoostClassifier(onnx_file=ONNX_FILE, config_file=CONFIG_FILE, n_jobs=CPUS)
    CONFIG.update(dict(use_image_morphings=False))
    data_module = PlanktonDataLoader(transform=None, return_filename=True)
    data_module.setup()

    dataloader = data_module.val_dataloader()
    radiomics_df = model.get_radiomics_from_dataloader(dataloader)