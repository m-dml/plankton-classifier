import unittest

import numpy as np

from src.utils import CONFIG
from src.utils.DataLoader import PlanktonDataLoader
from src.models.catboost_meta_classifier import BoostClassifier
import SimpleITK as sitk
import PIL

ONNX_FILE = "C:/Users/Tobias/Downloads/model_99917.onnx"
TEST_IMAGE = "C:\\Users\\Tobias\\PycharmProjects\\plankton-classifier\\data\\new_data\\4David\\M160\\Sorted\\Trochophora\\011219\\20191201_024326.836.0.png"


class TestBoostPipeline(unittest.TestCase):

    def setUp(self) -> None:
        config_file = "../tobis_config.yaml"
        CONFIG.update(config_file)

    def test_main_class_init(self):
        cl = BoostClassifier(onnx_file=ONNX_FILE)

        self.assertIsNotNone(cl)
        self.assertIsNone(cl.boost_model)
        self.assertIsNone(cl.classifier)
        self.assertFalse(cl.resnet_is_initialized)
        self.assertFalse(cl.catboost_is_initialized)

    def test_open_image(self):
        cl = BoostClassifier(onnx_file=ONNX_FILE)
        image = cl._open_image(TEST_IMAGE)

        self.assertIsNotNone(image)
        self.assertIsInstance(image, sitk.Image)

    def test_image_to_pil(self):
        cl = BoostClassifier(onnx_file=ONNX_FILE)
        image = cl._open_image(TEST_IMAGE)
        pil_image = cl._image_to_pil(image)

        self.assertIsNotNone(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(3, np.array(pil_image).shape[0])

    def test_calculate_radiomics(self):
        cl = BoostClassifier(onnx_file=ONNX_FILE)
        image = cl._open_image(TEST_IMAGE)
        radiomics = cl._calculate_radiomics(image)

        self.assertIsNotNone(radiomics)
        self.assertGreaterEqual(100, len(radiomics))

    def test_init_resnet_classifier(self):
        cl = BoostClassifier(onnx_file=ONNX_FILE)
        cl._init_resnet_classifier()

        self.assertTrue(cl.resnet_is_initialized)

    def test_make_predictions_with_resnet(self):
        cl = BoostClassifier(onnx_file=ONNX_FILE)
        cl._init_resnet_classifier()
        image = cl._open_image(TEST_IMAGE)
        pil_image = cl._image_to_pil(image)
        prediction = cl._make_predictions_with_resnet(pil_image)

        self.assertIsNotNone(prediction)
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual((17, ), prediction.shape)
        self.assertLessEqual(1, prediction.max())
        self.assertGreaterEqual(0, prediction.min())
        self.assertAlmostEqual(1, prediction.sum())

    def test_init_catboost_classifier(self):
        raise NotImplementedError

    def test_combine_radiomics_and_resnet_predictions(self):
        raise NotImplementedError

    def test_train_and_load_catboost(self):
        raise NotImplementedError

    def test_predict(self):
        raise NotImplementedError