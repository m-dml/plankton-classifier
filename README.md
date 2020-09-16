Classifier for Plankton data
============================

This repository is for working on a Classifier for images of plankton.
The major difficulty is accounting for the different image sizes.

You can download the data from the strand server (strand.fzg.local) at 
`/gpfs/work/machnitz/`


Importing the data into pytorch
-------------------------------

if you have the images in a folder structure like `/data/Train3_0` you can
use the `torchvision datasets` method to load the data in one line. 
Feel free to use the following snippet as an entry point:

```python
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

dataset = datasets.ImageFolder('/data/Train3_0')
train_split = int(len(dataset) * 0.9)
test_split = len(dataset) - train_split
train_dataset, test_dataset = random_split(dataset, [train_split, test_split])

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
...
```

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