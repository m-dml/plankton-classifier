#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:06:25 2022

@author: smalldatalooser
"""

import argparse
import os

import webdataset as wds

parser = argparse.ArgumentParser(description="Parameters for WebDataset creation", allow_abbrev=True)
parser.add_argument("srcpath", help="Path to parent directory containing image files/folders")
parser.add_argument("dstpath", help="Path to destination directory of the WebDataset")
parser.add_argument("--dst_prefix", default="", help="Prefix for shard file names")
parser.add_argument("--unsupervised", help="Will the Dataset be used for Pretraining? (Bool)", action="store_true")
parser.add_argument("--shard_size", default=1e9, help="Maximum size of Dataset Shard in Bytes")
parser.add_argument("--extension", default="png", help="image format/ file extension (png/jpg)")


def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def get_basenames(parent_dir):
    return getListOfFiles(parent_dir)


def create_unsupervised_dataset_from_folder_structure(
    srcpath, dstpath, dst_prefix, unsupervised, shard_size, extension
):
    basenames = getListOfFiles(os.path.expanduser(srcpath))
    basenames = [bsnam for bsnam in basenames if os.path.splitext(bsnam)[1] == extension]
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
