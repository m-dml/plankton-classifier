import argparse
import os

import webdataset as wds

parser = argparse.ArgumentParser(description="Parameters for WebDataset creation", allow_abbrev=True)
parser.add_argument("srcpath", help="Path to parent directory containing image files/folders")
parser.add_argument("dstpath", help="Path to destination directory of the WebDataset")
parser.add_argument("--dst_prefix", default="", help="Prefix for shard file names")
parser.add_argument("--unsupervised", help="Will the Dataset be used for Pretraining? (Bool)", action="store_true")
parser.add_argument("--shard_size", default=1e9, help="Maximum size of Dataset Shard in Bytes")


def get_list_of_files(dir_name):
    list_of_files = list()
    for (dirpath, dirnames, filenames) in os.walk(dir_name):
        list_of_files += [os.path.join(dirpath, file) for file in filenames]
    return list_of_files


def create_unsupervised_dataset_from_folder_structure(srcpath, dstpath, dst_prefix, unsupervised, shard_size):
    basenames = get_list_of_files(os.path.expanduser(srcpath))
    if not os.path.isdir(os.path.expanduser(dstpath)):
        os.makedirs(os.path.expanduser(dstpath))
    sink = wds.ShardWriter(os.path.join(dstpath, dst_prefix + "data_shard-%07d.tar"), maxsize=shard_size)
    for basename in basenames:
        with open(basename, "rb") as stream:
            image = stream.read()
        clss = os.path.basename(os.path.dirname(basename))
        if unsupervised:
            sample = {
                "__key__": os.path.splitext(basename)[0],
                "input.png": image,
            }
        else:
            sample = {"__key__": os.path.splitext(basename)[0], "input.png": image, "label.txt": clss}

        sink.write(sample)
    sink.close()
    return True


if __name__ == "__main__":
    args = parser.parse_args()
    create_unsupervised_dataset_from_folder_structure(**vars(args))
