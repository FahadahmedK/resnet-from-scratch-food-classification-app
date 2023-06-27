import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def transform(base_dir):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return datasets.ImageFolder(
        root=os.path.join(base_dir, "train"), transform=transform_train
    ), datasets.ImageFolder(
        root=os.path.join(base_dir, "test"), transform=transform_test
    )


def imgshow(image_tensor, title=None):
    image = image_tensor.numpy().permute((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    plt.show(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
