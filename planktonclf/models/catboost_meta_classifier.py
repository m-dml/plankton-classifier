import logging
import os

import SimpleITK as sitk
import numpy as np
import onnxruntime as ort
import pandas as pd
import radiomics
import yaml
from PIL import Image
from catboost import CatBoostClassifier
from joblib import Parallel, delayed, parallel_backend
from radiomics import featureextractor
from scipy.special import softmax
from torchvision import transforms
import socket
from tqdm import tqdm

from planktonclf.utils import CONFIG
from planktonclf.utils.DataLoader import PlanktonDataLoader
from planktonclf.utils.SquarePadTransform import SquarePad

radiomics.logger.setLevel(logging.ERROR)


class BoostClassifier:
    def __init__(self, onnx_file: str, config_file=None, n_jobs=1):

        self.update_config(config_file)

        self.onnx_file = onnx_file
        self.n_jobs = n_jobs
        self.boost_model = None
        self.classifier = None
        self.catboost_is_initialized = False
        self.resnet_is_initialized = False

    def update_config(self, config_file=None):
        if config_file:
            with open(os.path.abspath(config_file), "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            config_dict = dict()

        config_dict["batch_size"] = 1
        config_dict["shuffle_train_dataset"] = False
        # update values in the config class.
        CONFIG.update(config_dict)

    @staticmethod
    def _open_image(image_path: str) -> Image.Image:
        return Image.open(image_path)

    @staticmethod
    def _image_to_pil(image: sitk.Image) -> Image:
        array = sitk.GetArrayFromImage(image)
        array = np.moveaxis(array, -1, 0).astype(np.uint8)
        return Image.fromarray(array, "RGB")

    def _image_to_sitk(self, image: Image.Image) -> sitk.Image:
        array = np.array(image)
        sitk_image = sitk.GetImageFromArray(array)
        return sitk_image

    def transform_pil(self, image: Image.Image):
        transform = transforms.Compose([
                   SquarePad(),
                   transforms.Resize(size=[CONFIG.final_image_size, CONFIG.final_image_size]),
        ])
        return transform(image)

    def _init_resnet_classifier(self):
        # set up onnx
        options = ort.SessionOptions()
        options.inter_op_num_threads = self.n_jobs
        options.intra_op_num_threads = self.n_jobs

        # set up classifier
        self.classifier = ort.InferenceSession(self.onnx_file, sess_options=options)
        self.resnet_is_initialized = True

    def _make_predictions_with_resnet(self, image: Image) -> np.ndarray:
        """
        This should return the probabilites
        """
        if not self.resnet_is_initialized:
            self._init_resnet_classifier()

        input_name = self.classifier.get_inputs()[0].name
        output_name = self.classifier.get_outputs()[0].name

        transformed_image = self.transform_pil(image)
        model_input = np.expand_dims(np.array(transformed_image).astype(np.float32), 0)
        model_input = np.moveaxis(model_input, -1, 1)
        model_input[:, :3, :, :] = model_input[:, :3, :, :] / 255  # normalize the first three channels
        predictions = softmax(self.classifier.run([output_name], {input_name: model_input})[0])[0]
        return predictions

    def _create_training_data(self, resnet_predictions: pd.DataFrame, radiomics_df: pd.DataFrame):
        out_df = pd.concat([resnet_predictions.set_index("file_names"), radiomics_df.set_index("file_names")], axis=1)
        return out_df

    def get_radiomics_from_dataloader(self, dataloader, ray_backend=False) -> pd.DataFrame:
        radiomics_file = "radiomics_train_data.csv"
        if not os.path.isfile(radiomics_file):
            if ray_backend:
                print(f"I am {socket.gethostname()}")
                import ray
                from ray.util.joblib import register_ray  # noqa: E402
                ray.init(address=os.getenv("RAY_ADDRESS"), _redis_password=os.getenv("REDIS_PWD"))
                register_ray()
                with parallel_backend("ray"):
                    parallel_results = Parallel(verbose=7, n_jobs=-1)(
                        delayed(_do_predictions)(batch) for batch in tqdm(dataloader))
            else:
                parallel_results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                    delayed(_do_predictions)(batch) for batch in tqdm(dataloader))

            radiomics_list, parallel_labels, parallel_file_names = zip(*parallel_results)
            radiomics_df = pd.DataFrame([*radiomics_list])
            radiomics_df["file_names"] = parallel_file_names
            radiomics_df["labels"] = parallel_labels
            radiomics_df.to_csv("radiomics_train_data.csv", index=False)
            del radiomics_df

        radiomics_out_df = pd.read_csv(radiomics_file)
        return radiomics_out_df

    def get_resnet_predictions_from_dataloader(self, dataloader) -> pd.DataFrame:
        predictions = []
        file_names = []
        resnet_file = "resnet_train_data_for_catboost.csv"
        if not os.path.isfile(resnet_file):

            for batch in tqdm(dataloader):
                image, label, label_name, file_name = batch
                pil_image = transforms.ToPILImage()(image[0])

                predictions.append(self._make_predictions_with_resnet(pil_image))
                file_names.append(file_name[0])

            df = pd.DataFrame(predictions)
            df["file_names"] = file_names
            df.to_csv(resnet_file, index=False)
            del df

        df_out = pd.read_csv(resnet_file)
        return df_out

    def train_catboost_calssifier(self):
        data_module = PlanktonDataLoader(transform=None, return_filename=True)
        data_module.setup()
        dataloader = data_module.val_dataloader()
        radiomics_df = self.get_radiomics_from_dataloader(dataloader)

        dataloader = data_module.val_dataloader()
        resnet_df = self.get_resnet_predictions_from_dataloader(dataloader)

        train_df = self._create_training_data(radiomics_df, resnet_df)

        self.boost_model = CatBoostClassifier(iterations=1000, loss_function='MultiClass', learning_rate=0.03)
        self.boost_model.fit(train_df.drop(columns=["labels"]).values, train_df["labels"].values, verbose=False)

        self.boost_model.save_model("catboost_trained.bin")

        self.catboost_is_initialized = True

    def load_catboost_from_checkpoint(self, checkpoint_file: str):
        self.boost_model = CatBoostClassifier()
        self.boost_model.load_model(checkpoint_file)

        self.catboost_is_initialized = True

    def predict(self, image_file: str):
        if not self.catboost_is_initialized:
            raise RuntimeError("You have to initialize the Catboost before doing predictions. Do this by "
                               "either training it or loading a checkpoint")

        image = self._open_image(image_file)
        resnet_prediction = self._make_predictions_with_resnet(image)
        radiomics_list = _prepare_radiomics(_calculate_radiomics(image))

        resnet_labels = [str(x) for x in range(len(resnet_prediction))]

        df = pd.DataFrame(radiomics_list, index=range(1))
        df[resnet_labels] = resnet_prediction

        prediction = self.boost_model.predict(df.values)[0][0]
        return prediction


def _do_predictions(batch):
    image, label, label_name, file_name = batch
    pil_image = transforms.ToPILImage()(image[0])
    radiomics_list = _prepare_radiomics(_calculate_radiomics(pil_image))
    return radiomics_list, label_name[0], file_name[0]


def _prepare_radiomics(radiomics_dict):
    bad_keys = ["diagnostics_Configuration_Settings",
                "diagnostics_Configuration_EnabledImageTypes",
                "diagnostics_Image-original_Hash",
                "diagnostics_Image-original_Dimensionality",
                "diagnostics_Image-original_Spacing",
                "diagnostics_Mask-original_Hash"
                ]

    for key, value in radiomics_dict.items():
        if "diagnostics_Versions" in key:
            bad_keys.append(key)

        if key == "diagnostics_Image-original_Size":
            channel, width, height = value
            bad_keys.append(key)

        if isinstance(value, tuple):
            bad_keys.append(key)

        if isinstance(value, np.ndarray):
            if value.size > 1:
                bad_keys.append(key)

    for bad_key in bad_keys:
        try:
            radiomics_dict.pop(bad_key)
        except KeyError:
            continue

    radiomics_dict["original_number_of_channels"] = channel
    radiomics_dict["original_image_width"] = width
    radiomics_dict["original_image_height"] = height

    return radiomics_dict


def _calculate_radiomics(image: Image.Image):
    image_array = np.array(image)
    image_mask = np.zeros_like(image_array)

    image_array_median = np.median(image_array)

    image_mask[image_array >= image_array_median] = 1

    if image_mask.max() == 0:
        raise ValueError("Image does not contain anything.")

    image_sitk = sitk.GetImageFromArray(image_array)
    mask_sitk = sitk.GetImageFromArray(image_mask)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    result = extractor.execute(image_sitk, mask_sitk)
    return result
