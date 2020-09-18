from torchvision import datasets, transforms
from torch.utils.data import random_split

from torch.utils.data import DataLoader
import pytorch_lightning
from src.utils.Dataset import PlanktonDataset
from src.utils.Model import LitModel

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = PlanktonDataset('data/plankton_dataset/Training3_0', transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    valid_dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    trainer = pytorch_lightning.Trainer(gpus=1, max_epochs=5)
    model = LitModel()
    trainer.fit(model, train_dataloader, valid_dataloader)
