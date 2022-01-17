import copy
import faulthandler
import glob
import os
import platform
from typing import List

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from torchvision.transforms import Compose

from src.lib.config import Config, register_configs
from src.models.BaseModels import concat_feature_extractor_and_classifier
from src.utils import utils

faulthandler.enable()
# sometimes windows and matplotlib don't play well together. Therefore we have to configure win for plt:
if platform.system() == "Windows":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# register the structured configs:
register_configs()

# set up advanced logging:


@hydra.main(config_name="config", config_path="conf")
def main(cfg: Config):
    utils.extras(cfg)  # check if debug is activated and if so, change some trainer settings
    utils.set_log_levels(cfg.log_level)
    log = utils.get_logger(cfg.log_level)

    # Pretty print config using Rich library
    if cfg.print_config:
        utils.print_config(cfg, resolve=True)  # prints the complete hydra config to std-out

    torch.manual_seed(cfg.random_seed)  # set random seed
    pl.seed_everything(cfg.random_seed)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in cfg:
        for _, lg_conf in cfg["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Transformations
    train_transforms: Compose = hydra.utils.instantiate(cfg.datamodule.train_transforms)
    valid_transforms: Compose = hydra.utils.instantiate(cfg.datamodule.valid_transforms)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule,
        train_transforms=train_transforms,
        valid_transforms=valid_transforms,
        dataset=cfg.datamodule.dataset,
        is_ddp=cfg.strategy is not None,
    )
    datamodule.setup()  # manually set up the datamodule here, so an example batch can be drawn

    # generate example input array:
    for batch in datamodule.val_dataloader():
        example_input, _ = batch
        if isinstance(example_input, (tuple, list)):
            example_input = torch.stack(example_input).detach().cpu()
        break

    log.info(f"Size of one batch is: {example_input.element_size() * example_input.nelement() / 2**20} mb")

    is_in_simclr_mode = example_input.shape[0] == 2  # if first dimension is 2, then it is in simclr mode -> True
    log.info(f"Model is in simclr mode?: <{is_in_simclr_mode}>")

    # Init Lightning model
    log.info(f"Instantiating model <{cfg.lightning_module._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.lightning_module,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        feature_extractor=cfg.model.feature_extractor,
        classifier=cfg.model.classifier,
        loss=cfg.loss,
        class_labels=datamodule.unique_labels,
        all_labels=datamodule.all_labels,
        example_input_array=example_input.detach().cpu(),
        is_in_simclr_mode=is_in_simclr_mode,
        batch_size=cfg.datamodule.batch_size,
    )

    # load the state dict if one is provided (has to be provided for finetuning classifier in simclr):
    device = "cuda" if cfg.trainer.gpus > 0 else "cpu"
    if cfg.load_state_dict is not None:
        log.info(f"Loading model weights from {cfg.load_state_dict}")
        net = copy.deepcopy(model.model.cpu())
        # check state dict before loading:
        this_state_dict = model.model.state_dict().copy()
        len_old_state_dict = len(this_state_dict)
        log.info(f"Old state dict has {len_old_state_dict} entries.")
        try:
            new_state_dict = torch.load(cfg.load_state_dict, map_location=torch.device(device))
        except Exception as e:
            log.error()
            raise e
        for key in new_state_dict.keys():
            # make sure feature extractor weights are the same format:
            if key not in this_state_dict and not key.startswith("classifier."):
                # trigger complete traceback error if feature extractor weights are not the same
                net.load_state_dict(new_state_dict, strict=True)

        keys_to_drop = list(this_state_dict.keys())[-2:]
        [new_state_dict.pop(key_to_drop) for key_to_drop in keys_to_drop]
        missing_keys, unexpected_keys = net.load_state_dict(new_state_dict, strict=False)
        log.warning(f"Missing keys: {missing_keys}")
        log.warning(f"Unexpected keys: {unexpected_keys}")
        state_dict_error_count = 0
        for state_key, state in net.state_dict().items():
            if this_state_dict[state_key].allclose(state, atol=1e-12, rtol=1e-12):
                log.error(
                    f"Loaded state dict params for layer '{state_key}' are same as random initialized one ("
                    f"Might be due to caching, if you just restarted the same model twice!)"
                )
                state_dict_error_count += 1
        if state_dict_error_count > 0:
            log.warning(
                f"{state_dict_error_count} state entries are the same after init. "
                f"(From a total of {len_old_state_dict} items)"
            )

        model.feature_extractor = copy.deepcopy(net.feature_extractor.to(device))
        model.classifier = copy.deepcopy(net.classifier.to(device))
        model.model = concat_feature_extractor_and_classifier(model.feature_extractor, model.classifier)
        del net
        log.info(f"Successfully loaded model weights from {cfg.load_state_dict}")

        state_folder = os.path.split(cfg.load_state_dict)[0]
        temperature_files = glob.glob(os.path.join(state_folder, "temperature*.tensor"))
        if (len(temperature_files) > 0) and cfg.lightning_module.temperature_scale:
            step_number = os.path.split(cfg.load_state_dict)[1].split("complete_model_")[-1].split(".weights")[0]
            temperature_file = os.path.join(state_folder, f"temperatures_{step_number}.tensor")
            model.temperatures = torch.load(temperature_file)

    # freeze the weights of the feature extractor to only train the classifier
    if cfg.lightning_module.freeze_feature_extractor:
        log.info("Freezing the weights of the feature extractor")
        net = copy.deepcopy(model.model.cpu())
        for name, module in net.named_modules():
            if name.startswith("feature_extractor"):
                for param in module.parameters():
                    param.requires_grad = False
        model.feature_extractor = copy.deepcopy(net.feature_extractor.to(device))
        model.classifier = copy.deepcopy(net.classifier.to(device))
        model.model = concat_feature_extractor_and_classifier(model.feature_extractor, model.classifier)
        del net

    # log hparam metrics to tensorboard:
    log.info("Logging hparams to tensorboard")
    hydra_params = utils.log_hyperparameters(config=cfg, model=model)
    for this_logger in logger:
        if "tensorboard" in str(this_logger):
            log.info("Add hparams to tensorboard")
            this_logger.log_hyperparams(hydra_params, {"hp/loss": 0, "hp/accuracy": 0, "hp/epoch": 0})
        else:
            this_logger.log_hyperparams(hydra_params)

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters to lightning!")
    model.hydra_params = hydra_params

    # Init Trainer:
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = instantiate(
        cfg.trainer, strategy=cfg.strategy, logger=logger, callbacks=callbacks, _convert_="partial"
    )

    # if activated in the config, start the pytorch lightning automatic batch-size and lr tuning process
    if cfg.auto_tune:
        log.info("Starting tuning the model")
        trainer.tune(model, datamodule)

    log.info("Starting training")
    trainer.fit(model, datamodule)  # the actual training of the NN

    # Print path to best checkpoint
    if trainer.checkpoint_callback.best_model_path is not None:
        log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    return trainer.callback_metrics["hp/accuracy"].item()


if __name__ == "__main__":
    log = utils.get_logger()
    log.info("Starting python script")
    try:
        main()
    except Exception as e:
        log.exception("!!! Model Failed !!!")
        raise e
