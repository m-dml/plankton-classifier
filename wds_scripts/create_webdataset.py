import argparse
import logging
import os
import sys

import webdataset as wds

parser = argparse.ArgumentParser(description="Parameters for WebDataset creation", allow_abbrev=True)
parser.add_argument("src_path", help="Path to parent directory containing image files/folders")
parser.add_argument("dst_path", help="Path to destination directory of the WebDataset")
parser.add_argument("--dst_prefix", default="", help="Prefix for shard file names")
parser.add_argument("--unsupervised", help="Will the Dataset be used for Pretraining? (Bool)", action="store_true")
parser.add_argument("--shard_size", default=1e9, help="Maximum size of Dataset Shard in Bytes", type=float)
parser.add_argument("--verbose", action="store_true", help="Prints additional information")
parser.add_argument("--extension", default="png", help="image format/ file extension (png/jpg)")


def create_unsupervised_dataset_from_folder_structure(
    src_path, dst_path, dst_prefix, unsupervised, extension, shard_size, *_, **__
):
    if not os.path.isdir(os.path.expanduser(dst_path)):
        os.makedirs(os.path.expanduser(dst_path))
        logging.info("Created Directory: {}".format(dst_path))

    sink = wds.ShardWriter(os.path.join(dst_path, f".{dst_prefix}_data_shard-%07d.tar"), maxsize=shard_size)
    for (dir_path, _, filenames) in os.walk(src_path):
        list_of_files = [os.path.join(dir_path, file) for file in filenames]
        logging.info("Processing folder: {}".format(dir_path))

        list_of_files = [
            basename
            for basename in list_of_files
            if os.path.splitext(basename)[1].lower().strip() == "." + extension.lower().strip()
        ]
        for file in list_of_files:
            with open(file, "rb") as stream:
                image = stream.read()
            clss = os.path.basename(os.path.dirname(file))
            if unsupervised:
                sample = {
                    "__key__": os.path.splitext(file)[0],
                    "input.png": image,
                }
            else:
                sample = {"__key__": os.path.splitext(file)[0], "input.png": image, "label.txt": clss}

            sink.write(sample)
    sink.close()


if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("create_webdataset.log")
            ]
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )

    create_unsupervised_dataset_from_folder_structure(**vars(args))
