import logging
import os

import SimpleITK as sitk
import catboost
import numpy as np
import onnxruntime as ort
import pandas as pd
import radiomics
import yaml
from PIL import Image
from catboost import CatBoostClassifier
from radiomics import featureextractor
from scipy.special import softmax
from torchvision import transforms

from src.utils import CONFIG
from src.utils.DataLoader import PlanktonDataLoader
from src.utils.SquarePadTransform import SquarePad

radiomics.logger.setLevel(logging.ERROR)


class BoostClassifier:
    def __init__(self, onnx_file: str, config_file=None):

        if config_file is not None:
            self.update_config(config_file)

        self.onnx_file = onnx_file

        self.boost_model = None
        self.classifier = None
        self.catboost_is_initialized = False
        self.resnet_is_initialized = False

    @staticmethod
    def update_config(config_file):
        with open(os.path.abspath(config_file), "r") as f:
            config_dict = yaml.safe_load(f)

        config_dict["batch_size"] = 1
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

    @staticmethod
    def _calculate_radiomics(image: Image.Image):
        image_array = np.array(image)
        image_mask = np.zeros_like(image_array)

        image_mask[image_array >= 10] = 1

        if image_mask.max() == 0:
            raise ValueError("Image does not contain anything.")

        image_sitk = sitk.GetImageFromArray(image_array)
        mask_sitk = sitk.GetImageFromArray(image_mask)

        extractor = featureextractor.RadiomicsFeatureExtractor()
        result = extractor.execute(image_sitk, mask_sitk)
        return result

    def _init_resnet_classifier(self):
        # set up onnx
        options = ort.SessionOptions()
        options.inter_op_num_threads = 12
        options.intra_op_num_threads = 12

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
        predictions = softmax(self.classifier.run([output_name], {input_name: model_input})[0])[0]
        return predictions

    def _combine_radiomics_and_resnet_predictions(self, radiomics, resnet_predictions, label) -> pd.DataFrame:
        resnet_labels = [str(x) for x in range(len(resnet_predictions))]
        df = pd.DataFrame(radiomics)
        df["label"] = label
        df[resnet_labels] = resnet_predictions
        return df

    def _create_training_data(self, resnet_predictions: list, radiomics: list, labels: list):

        resnet_labels = [str(x) for x in range(len(resnet_predictions[0]))]

        df = pd.DataFrame([*radiomics])
        df["labels"] = labels
        df[resnet_labels] = resnet_predictions
        return df

    def train_catboost_calssifier(self):
        """
        :param dataloader: a torch dataloader
        """

        data_module = PlanktonDataLoader(transform=None)
        data_module.setup()
        dataloader = data_module.train_dataloader()

        predictions = []
        radiomics = []
        labels = []

        for batch in dataloader:

            image, label, label_name = batch
            pil_image = transforms.ToPILImage()(image[0])

            predictions.append(self._make_predictions_with_resnet(pil_image))
            radiomics.append(self._prepare_radiomics(self._calculate_radiomics(pil_image)))
            labels.append(label.cpu().numpy()[0][0])

        train_df = self._create_training_data(predictions, radiomics, labels)

        self.boost_model = CatBoostClassifier()
        self.boost_model.fit(train_df.drop(columns=["labels"]).values, train_df["labels"].values, verbose=False)
        self.boost_model.save_model("catboost_trained.bin")

        self.catboost_is_initialized = True

    def _prepare_radiomics(self, radiomics):

        bad_keys = ["diagnostics_Configuration_Settings",
                    "diagnostics_Configuration_EnabledImageTypes",
                    "diagnostics_Image-original_Hash",
                    "diagnostics_Image-original_Dimensionality",
                    "diagnostics_Image-original_Spacing",
                    "diagnostics_Mask-original_Hash"
                    ]

        for key, value in radiomics.items():
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
                radiomics.pop(bad_key)
            except KeyError:
                continue

        radiomics["original_number_of_channels"] = channel
        radiomics["original_image_width"] = width
        radiomics["original_image_height"] = height

        return radiomics

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
        radiomics = self._prepare_radiomics(self._calculate_radiomics(image))

        resnet_labels = [str(x) for x in range(len(resnet_prediction))]

        df = pd.DataFrame(radiomics, index=range(1))
        df[resnet_labels] = resnet_prediction

        prediction = self.boost_model.predict(df.values)
        return prediction
