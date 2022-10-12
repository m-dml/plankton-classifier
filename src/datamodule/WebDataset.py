import torch
import webdataset
from torchvision.transforms import transforms


class BaseWebDataset(webdataset):
    def __init__(self, integer_labels, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.integer_labels = integer_labels
        self.transform = transform


class FinetuneWebDataset(BaseWebDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for sample in super().__iter__():
            image, label_name = sample["input"], sample["label"]

            if self.transforms:
                image = self.transforms(image)

            if not isinstance(image, torch.Tensor):
                image = transforms.ToTensor()(image)

            yield image, (torch.Tensor([self.integer_labels[label_name]]), label_name)


class PretrainWebDataset(BaseWebDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for sample in super().__iter__():
            image = sample["input"]
            image_copy = image.copy()

            if self.transforms:
                image = self.transform(image)
                image_copy = self.transform(image_copy)
            else:
                raise ValueError("Transforms must be defined for pretraining")

            yield (image, image_copy), (torch.tensor([0]))
