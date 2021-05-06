import numpy as np
import catboost
from catboost import CatBoostClassifier
import onnxruntime as ort
import SimpleITK as sitk
from radiomics import featureextractor
from PIL import Image
import torch
from torchvision import transforms

from src.utils.SquarePadTransform import SquarePad
from src.utils import CONFIG
from scipy.special import softmax


class BoostClassifier:
    def __init__(self, onnx_file: str):
        self.onnx_file = onnx_file

        self.boost_model = None
        self.classifier = None
        self.catboost_is_initialized = False
        self.resnet_is_initialized = False

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

        image_mask[image_array > 5] = 1

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

    def _init_catboost_classifier(self):
        pass

    def _combine_radiomics_and_resnet_predictions(self, radiomics, resnet_predictions):
        pass

    def train_catboost_calssifier(self, dataloader):
        """
        :param dataloader: a torch dataloader
        """
        for batch in dataloader:
            images, labels, label_names = batch
            predictions = self._make_predictions_with_resnet(images)

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
        resnet_prediction = self._make_predictions_with_resnet(self._image_to_pil(image))
        radiomics = self._calculate_radiomics(image)

        data = self._combine_radiomics_and_resnet_predictions(radiomics, resnet_prediction)

        prediction = catboost.predict(self.boost_model, data)
        return prediction
