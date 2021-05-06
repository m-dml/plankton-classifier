import numpy as np
import catboost
from catboost import CatBoostClassifier
import onnxruntime as ort
import SimpleITK as sitk
from radiomics import featureextractor
from PIL import Image


class CatboostClassifier:
    def __init__(self, onnx_file: str):
        self.onnx_file = onnx_file

        self.boost_model = None
        self.classifier = None
        self.catboost_is_initialized = False
        self.resnet_is_initialized = False

    @staticmethod
    def _open_image(image_path):
        return sitk.ReadImage(image_path)

    @staticmethod
    def _image_to_pil(image):
        array = sitk.GetArrayFromImage(image)
        return Image.fromarray()

    @staticmethod
    def _calculate_radiomics(image):
        array = sitk.GetArrayFromImage(image)
        mask_array = np.zeros_like(array)
        mask_array[array > 5] = 1
        mask = sitk.GetImageFromArray(mask_array)
        extractor = featureextractor.RadiomicsFeatureExtractor()
        result = extractor.execute(image, mask)
        return result

    def _init_resnet_classifier(self):
        # set up onnx
        options = ort.SessionOptions()
        options.inter_op_num_threads = 12
        options.intra_op_num_threads = 12

        # set up classifier
        self.classifier = ort.InferenceSession(self.onnx_file, sess_options=options)
        self.resnet_is_initialized = True

    def _make_predictions_with_resnet(self, images):
        """
        This should return the probabilites
        """
        if not self.resnet_is_initialized:
            self._init_resnet_classifier()

        input_name = self.classifier.get_inputs()[0].name
        output_name = self.classifier.get_outputs()[0].name

        model_input = images.cpu().numpy()
        predictions = self.classifier.run([output_name], {input_name: model_input})[0]
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
        resnet_prediction = self._make_predictions_with_resnet(image)
        prediction = catboost.predict(self.boost_model, image)
        return prediction
