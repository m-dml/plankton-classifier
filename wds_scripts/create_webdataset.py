import argparse
import asyncio
import logging
import os
import sys

import webdataset as wds

parser = argparse.ArgumentParser(description="Parameters for WebDataset creation", allow_abbrev=True)
parser.add_argument("src_path", help="Path to parent directory containing image files/folders")
parser.add_argument("dst_path", help="Path to destination directory of the WebDataset")
parser.add_argument("--dst_prefix", default="", help="Prefix for shard file names")
parser.add_argument("--unsupervised", help="Will the Dataset be used for Pretraining? (Bool)", action="store_true")
parser.add_argument("--shard_size", default=1e9, help="Maximum size of Dataset Shard in Bytes")
parser.add_argument("--verbose", action="store_true", help="Prints additional information")
parser.add_argument("--extension", default="png", help="image format/ file extension (png/jpg)")


async def create_unsupervised_dataset_from_folder_structure(
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


async def apply_command(arguments):
    proc = await create_unsupervised_dataset_from_folder_structure(**arguments)
    return proc


async def gather_with_concurrency(n, *tasks):
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))


async def prepare_command(commands):
    coroutines = [apply_command(command) for command in commands]
    await gather_with_concurrency(8, *coroutines)


def main(logger, *args, **kwargs):
    coroutines = []
    for subfolder in [f.path for f in os.scandir(kwargs["src_path"]) if f.is_dir()]:
        logging.info("Processing folder: {}".format(subfolder))
        these_kwargs = kwargs.copy()
        these_kwargs["src_path"] = os.path.join(kwargs["src_path"], os.path.basename(subfolder))
        these_kwargs["dst_path"] = os.path.join(kwargs["dst_path"], os.path.basename(subfolder))
        logging.debug("src_path is now: {}".format(these_kwargs["src_path"]))
        logging.debug("dst_prefix is now: {}".format(these_kwargs["dst_path"]))
        if not os.path.exists(these_kwargs["dst_path"]):
            os.makedirs(these_kwargs["dst_path"])
        coroutines.append(these_kwargs)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(prepare_command(coroutines))


if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout)
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
    logger = logging.getLogger()
    main(**vars(args), logger=logger)
