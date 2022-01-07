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


pre-commit
---------
Just use `pre-commit run --all-files` at the top level of this repo, to
let precommit handle the files.

Quick refresher on git and github
---------------------------------

1. Install git.

2. Create a github account.

3. `cd` into the directory where you want to have this code and clone this repository.

    ```bash
    git clone https://github.com/m-dml/plankton-classifier .

    cd plankton-classifier
    ```

4. Create a branch for you to work on. Replace `branch_name` with your name or some else
    unique name.

    ```bash
    git checkout -b branch_name
    ```

5. Do some code changes. Create an awesome classifier!

6. Check the project status with `git status`.

7. Add all files you want to upload and make a git commit:
    ```bash
   git add file1.py
   git add file2.py
   ...

   git commit -m "A commit message describing your code changes"
    ```
   Please do not upload the camera-images!

8. Push your changes to github
    ```bash
    git push origin branch_name
    ```
