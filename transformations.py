import torch
from torchvision import transforms


def transform_function(transform_type: str="default"):
    all_transform_types = ["default", "composed", "random"]
    if transform_type not in all_transform_types:
        transform_type = "default"

    if transform_type == "default":
        transform = transforms.Compose([
                                transforms.Pad(CONFIG.final_image_size),
                                transforms.CenterCrop([CONFIG.final_image_size, CONFIG.final_image_size]),
                                transforms.ToTensor()])
    elif transform_type == "composed":
        transform = transforms.Compose([transforms.Pad(), transforms.CenterCrop(10), transforms.ToTensor()])
    elif transform_type == "random":
        # Apply transforms out of a list of transforms randomly for a given probability
        transform = transforms.RandomApply(
            torch.nn.ModuleList([transforms.RandomCrop(), transforms.RandomRotation(degrees=[-10., 10.])]), p=0.3)
        transform = torch.jit.script(transform)

    return transform

print(transform_function("default"))