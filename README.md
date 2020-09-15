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
