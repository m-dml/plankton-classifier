Classifier for Plankton data
============================

This repository is for working on a Classifier for images of plankton.
The major difficulty is accounting for the different image sizes.

You can download the data from the strand server (strand.fzg.local) at
`/gpfs/work/machnitz/plankton-dataset`


Using this repo
-------------------------------
To run the vanilla model just run `main.py`. A conda environment which works with this project can be installed
with `conda env create -f environment.yaml`.
You can get a list of all options using `python main.py --help`.
For more information on configuration see the hydra docs: https://hydra.cc/

This repo contains a submodule, so after cloning it you have to also clone the submodule:

```bash
git clone https://github.com/m-dml/plankton-classifier
cd plankton-classifier
git submodule init
git submodule update
```

Inference with a trained model
-------------------------------

What you need:
- The checkpoint file of the trained model `some_file.ckpt`
- The integer-to-labelname file `class_labels.json` that was created during training of the model
- A folder containing images of plankton to be classified. This folder is allowed to have subfolders.

1. Install the environment with ``conda env create -f environment.yaml``.
2. Activate the environment with ``conda activate plankton``.
3. Make sure the checkpoint and the class-label file into one folder.
4. Run the inference script 
   1. Run it locally with: ``python main.py +experiment=inference/inference load_state_dict=some_file.ckpt 
   output_dir_base_path=/path/to/store/outputs/ ``
      - on Windows make sure to also always add ``datamodule.num_workers=0``
      - This command assumes that you have a GPU and are running the program locally. For more control clone the
      configuration file ``conf/experiment/inference/inference.yaml`` and make changes accordingly.

   2. To run the inference script on a slurm cluster add ``-m`` at the end of the command. Make sure that the right
   trainer and hydra-launcher are selected in your script.

pre-commit
---------
Just use `pre-commit run --all-files` at the top level of this repo, to
let precommit handle the files.
