Classifier for Plankton data
============================

This repository is for working on a Classifier for images of plankton.
The major difficulty is accounting for the different image sizes.

You can download the data from the strand server (strand.fzg.local) at
`/gpfs/work/machnitz/plankton-dataset`


Using this repo
-------------------------------
To run the vanilla model just run `main.py`. A conda environment which works with this project can be installed
with `conda env create -f environment.yaml`. To adjust hyperparameters make yourself a copy of the `default_config.yaml`
file and then run the program with `main.py -f my_config.yaml`.
Also, please adjust the account name in the `train_on_strand.sh`.


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
