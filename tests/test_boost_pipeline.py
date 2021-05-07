import logging
import os
import unittest

import PIL
import SimpleITK as sitk
import numpy as np
import pandas as pd

from planktonclf.models.catboost_meta_classifier import BoostClassifier, _calculate_radiomics
from planktonclf.utils import CONFIG
from planktonclf.utils.DataLoader import PlanktonDataLoader

ONNX_FILE = "C:/Users/Tobias/Downloads/model_99917.onnx"
TEST_IMAGE = "C:\\Users\\Tobias\\PycharmProjects\\plankton-classifier\\data\\new_data\\4David\\M160\\Sorted\\" \
             "Trochophora\\011219\\20191201_024326.836.0.png"


class TestBoostPipeline(unittest.TestCase):

    def setUp(self) -> None:
        config_file = "../tobis_config.yaml"
        CONFIG.update(config_file)

        if CONFIG.debug_mode:
            logging.basicConfig(level=logging.DEBUG, format='%(name)s %(funcName)s %(levelname)s %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format='%(name)s %(funcName)s %(levelname)s %(message)s')

    def tearDown(self) -> None:
        if os.path.isfile("radiomics_train_data.csv"):
            os.remove("radiomics_train_data.csv")
        if os.path.isfile("resnet_train_data_for_catboost.csv"):
            os.remove("resnet_train_data_for_catboost.csv")

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
        self.assertIsInstance(image, PIL.Image.Image)

    def test_image_to_pil(self):
        cl = BoostClassifier(onnx_file=ONNX_FILE)
        image = cl._open_image(TEST_IMAGE)
        sitk_image = cl._image_to_sitk(image)
        pil_image = cl._image_to_pil(sitk_image)

        self.assertIsNotNone(pil_image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(3, np.array(pil_image).shape[0])

    def test_image_to_sitk(self):
        cl = BoostClassifier(onnx_file=ONNX_FILE)
        image = cl._open_image(TEST_IMAGE)
        sitk_image = cl._image_to_sitk(image)

        self.assertIsNotNone(image)
        self.assertIsInstance(sitk_image, sitk.Image)
        self.assertEqual(3, sitk.GetArrayFromImage(sitk_image).shape[-1])

    def test_calculate_radiomics(self):
        cl = BoostClassifier(onnx_file=ONNX_FILE)
        image = cl._open_image(TEST_IMAGE)
        radiomics = _calculate_radiomics(image)

        self.assertIsNotNone(radiomics)
        self.assertGreaterEqual(len(radiomics), 100)

    def test_init_resnet_classifier(self):
        cl = BoostClassifier(onnx_file=ONNX_FILE)
        cl._init_resnet_classifier()

        self.assertTrue(cl.resnet_is_initialized)

    def test_make_predictions_with_resnet(self):
        cl = BoostClassifier(onnx_file=ONNX_FILE)
        cl._init_resnet_classifier()
        image = cl._open_image(TEST_IMAGE)
        prediction = cl._make_predictions_with_resnet(image)

        self.assertIsNotNone(prediction)
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual((17, ), prediction.shape)
        self.assertLessEqual(1, prediction.max())
        self.assertGreaterEqual(0, prediction.min())
        self.assertAlmostEqual(1, prediction.sum(), places=4)

    def test_create_radiomics_csv(self):
        cl = BoostClassifier(onnx_file=ONNX_FILE, n_jobs=2)
        cl._init_resnet_classifier()

        data_module = PlanktonDataLoader(transform=None, return_filename=True)
        data_module.setup()
        dataloader = data_module.val_dataloader()

        radiomics_df = cl.get_radiomics_from_dataloader(dataloader)
        self.assertIsInstance(radiomics_df, pd.DataFrame)
        self.assertEqual(len(dataloader), len(radiomics_df))
        self.assertTrue("png" in radiomics_df["file_names"][0].lower())

    def test_create_resnet_csv(self):
        cl = BoostClassifier(onnx_file=ONNX_FILE, n_jobs=2)
        cl._init_resnet_classifier()

        data_module = PlanktonDataLoader(transform=None, return_filename=True)
        data_module.setup()
        dataloader = data_module.val_dataloader()

        resnet_df = cl.get_resnet_predictions_from_dataloader(dataloader)
        self.assertIsInstance(resnet_df, pd.DataFrame)
        self.assertEqual(len(dataloader), len(resnet_df))
        self.assertTrue("png" in resnet_df["file_names"][0].lower())

    def test_train_and_load_catboost(self):
        # this test can take a very long time, so please exclude for quick tests:
        cl = BoostClassifier(onnx_file=ONNX_FILE, config_file="../tobis_config.yaml", n_jobs=6)
        CONFIG.update(dict(validation_split=0.00016))
        cl._init_resnet_classifier()
        cl.train_catboost_calssifier()

        self.assertTrue(cl.catboost_is_initialized)

    def test_predict(self):
        cl = BoostClassifier(onnx_file=ONNX_FILE, config_file="../tobis_config.yaml")
        cl._init_resnet_classifier()
        data_module = PlanktonDataLoader(transform=None, return_filename=True)
        data_module.setup()
        cl.load_catboost_from_checkpoint(checkpoint_file="catboost_trained.bin")
        prediction = cl.predict(TEST_IMAGE)
        print(prediction)

        self.assertTrue(cl.catboost_is_initialized)
        self.assertIsInstance(prediction, str)
        self.assertTrue(prediction in data_module.unique_labels)


if __name__ == "__main__":
    unittest.main()
