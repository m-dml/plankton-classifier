import torch
from torchvision import transforms


def transform_function(transform_type: str = "default", final_image_size=100):

    all_transform_types = ["default", "composed", "random"]
    if transform_type not in all_transform_types:
        transform_type = "default"

    if transform_type == "default":
        transform = transforms.Compose(
            [transforms.Pad(final_image_size), transforms.CenterCrop([final_image_size, final_image_size]),
             transforms.ToTensor()])
    elif transform_type == "composed":
        transform = transforms.Compose([transforms.Pad(), transforms.CenterCrop(10), transforms.ToTensor()])
    elif transform_type == "random":
        # Apply transforms out of a list of transforms randomly for a given probability
        transform = transforms.RandomApply(
            torch.nn.ModuleList([transforms.RandomCrop(), transforms.RandomRotation(degrees=[-10., 10.])]), p=0.3)
        transform = torch.jit.script(transform)
    elif transform_type == "random_rotations":
        transform = transforms.RandomRotation(degrees=[-10., 10.])
    return transform
