import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image
import click
from utils import load_yaml_config


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        label = image_path.split(os.path.sep)[-2]

        if self.transform:
            image = self.transform(image)

        return image, label

    def _get_image_paths(self):
        image_paths = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_paths.append(os.path.join(root, file))
        return image_paths


def preprocess(data_dir, output_dir, batch_size):
    train_output_dir = os.path.join(output_dir, "train")
    test_output_dir = os.path.join(output_dir, "test")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

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

    train_dataset = ImageDataset(
        data_dir=os.path.join(data_dir, "train"), transform=transform_train
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    for images, labels in train_loader:
        for i, (image, label) in enumerate(zip(images, labels)):
            image_path = os.path.join(train_output_dir, label)
            os.makedirs(image_path, exist_ok=True)
            save_image(image, os.path.join(image_path, f"{label}_{i}.jpg"))

    test_dataset = ImageDataset(
        data_dir=os.path.join(data_dir, "test"), transform=transform_test
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    for images, labels in test_loader:
        for i, (image, label) in enumerate(zip(images, labels)):
            image_path = os.path.join(test_output_dir, label, f"{label}_{i}.jpg")
            os.makedirs(image_path, exist_ok=True)
            save_image(image, os.path.join(image_path, f"{label}_{i}.jpg"))
    return output_dir


def imgshow(image_tensor, title=None):
    image = image_tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image * std) + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


@click.command()
# @click.option("--data_dir", default="data/raw", help="Path to raw data")
# @click.option("--output_dir", default="data/processed", help="Path to processed data")
@click.argument(
    "config_path", type=click.Path(exists=True))
def main(config_path):
    config = load_yaml_config(config_path)
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    batch_size = config["batch_size"]
    preprocess(data_dir, output_dir, batch_size)


if __name__ == "__main__":
    main()
