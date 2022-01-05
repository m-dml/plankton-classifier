import unittest

from src.datamodule.DataLoader import PlanktonDataLoader, PlanktonDataSetSimCLR


class TestPlanktonDataLoader(unittest.TestCase):
    def test_recursive_img_folder_recognition(self):
        dataloader = PlanktonDataLoader(
            excluded_labels=[],
            batch_size=4,
            num_workers=0,
            train_split=0.8,  # The fraction size of the training data
            validation_split=0.1,  # The fraction size of the validation data (rest ist test)
            shuffle_train_dataset=True,  # whether to shuffle the train dataset (bool)
            shuffle_validation_dataset=True,
            shuffle_test_dataset=False,
            preload_dataset=False,
            use_planktonnet_data=False,
            use_klas_data=False,
            use_canadian_data=False,
            super_classes=None,
            oversample_data=False,
            klas_data_path="",
            planktonnet_data_path="",
            canadian_data_path="",
            random_seed=42,
            train_transforms=None,
            valid_transforms=None,
            dataset=PlanktonDataSetSimCLR,
            reduce_data=False,
            pin_memory=False,
            unlabeled_files_to_append=None,
        )

        files = dataloader.add_all_images_from_all_subdirectories(
            folder="C:/Users/Tobias/PycharmProjects/plankton-classifier/data"
        )

        print(f"len(files = {len(files)})")
        self.assertGreater(len(files), 10000)
