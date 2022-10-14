import argparse
import asyncio
import glob
import os
import tarfile

from tqdm import tqdm


def extract(input_file, output_path, pbar=None):
    tar = tarfile.open(input_file)
    tar.extractall(path=output_path)
    tar.close()
    if pbar:
        pbar.update(1)


async def custom_system_command(command, pbar=None):
    command_list = command
    proc = await asyncio.create_subprocess_exec(*command_list)
    returncode = await proc.wait()
    if pbar:
        pbar.update(1)

    file = command[2]
    os.remove(file)

    return returncode


async def gather_with_concurrency(n, *tasks):
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))


async def prepare_system_command(commands):
    coroutines = [custom_system_command(command) for command in commands]
    await gather_with_concurrency(8, *coroutines)


def main(_arg_list):
    commands = []
    for config in sorted(_arg_list):

        commands.append(["python",
                         config["src_path"],
                         config["dst_path"],
                         "--dst_prefix", config["dst_prefix"],
                         "--unsupervised",
                         "--verbose",
                         "--shard_size", config["shard_size"],
                         "--extension", config["extension"]])

    loop = asyncio.get_event_loop()
    loop.run_until_complete(prepare_system_command(commands))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for WebDataset creation", allow_abbrev=True)
    parser.add_argument("src_path", help="Path to parent directory containing image files/folders")
    parser.add_argument("dst_path", help="Path to destination directory of the WebDataset")
    parser.add_argument("--dst_prefix", default="", help="Prefix for shard file names")
    parser.add_argument("--unsupervised", help="Will the Dataset be used for Pretraining? (Bool)", action="store_true")
    parser.add_argument("--shard_size", default=1e9, help="Maximum size of Dataset Shard in Bytes")
    parser.add_argument("--verbose", action="store_true", help="Prints additional information")
    parser.add_argument("--extension", default="png", help="image format/ file extension (png/jpg)")
    args = parser.parse_args()

    print(args["src_path"])

    folders = [x for x in glob.glob(os.path.join(args["src_path"], "*")) if os.path.isdir(x)]
    arg_list = []
    for folder in tqdm(folders):
        these_args = args.copy()
        these_args["src_path"] = folder
        these_args["dst_path"] = os.path.join(args["dst_path"], os.path.basename(folder))
        arg_list.append(these_args)

    main(arg_list)
