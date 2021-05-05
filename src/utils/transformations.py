import torch
from torchvision import transforms


def get_transforms_from_config(selected_transforms: list = ["Pad_CenterCrop", "ToTensor"], final_image_size=100):
    if selected_transforms == []:
        selected_transforms.append("Pad_CenterCrop")
    list_of_transforms = []
    for transform_type in selected_transforms:
        if transform_type == "Pad_CenterCrop":
            list_of_transforms.append(transforms.Pad(final_image_size))
            list_of_transforms.append(transforms.CenterCrop([final_image_size, final_image_size]))
        elif transform_type == "Random_Rotations":
            list_of_transforms.append(transforms.RandomRotation(degrees=[-180., 180.]))
        elif transform_type == "ToTensor":
            list_of_transforms.append(transforms.ToTensor())
        else:
            raise ValueError("You chose an invalid argument for transform.")
    transform = transforms.Compose(list_of_transforms)
    return transform
