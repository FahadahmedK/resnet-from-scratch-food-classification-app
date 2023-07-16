import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image
import click
from src.utils import load_yaml_config
from src.utils import label_to_idx


class ImageDataset(Dataset):
    """This class is used to load the images from the data directory and apply the transformations on them."""

    def __init__(self, data_dir, transform=None):

        """
        data_dir: Path to data
        transform: Transformations to be applied on the images
        """

        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()
        self.labels = {label.name for label in Path(self.data_dir).glob('*')}
        self.label_map = open(Path(self.data_dir).parents[1] / 'meta/label_map.json', 'r')
        with open(Path(self.data_dir).parents[1] / 'meta/label_map.json', 'r') as f:
            self.label_map = json.load(f)


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        label = image_path.parent.name

        if self.transform:
            image = self.transform(image)
        return image, label_to_idx(label, label_map=self.label_map), label, os.path.split(image_path)[1]

    def _get_image_paths(self):
        image_paths = []
        for image in Path(self.data_dir).glob('**/*.jpg'):
            image_paths.append(image)
        # for root, dirs, files in os.walk(self.data_dir):
        #    for file in files:
        #        if file.endswith(".jpg") or file.endswith(".png"):
        #            image_paths.append(os.path.join(root, file))
        return image_paths


def preprocess_images(data_dir: str, output_dir: str, batch_size: int = 1, is_train=True, save=True):
    """ This function is used to preprocess the images and save them in the output directory. """

    """
    data_dir: Path to data
    output_dir: Path to save the preprocessed images
    is_train: If True, preprocess the training data else preprocess the test data
    batch_size: Batch size to be used for preprocessing
    """

    if is_train:
        mode = 'train'
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        mode = 'test'
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    try:
        dataset = ImageDataset(data_dir=os.path.join(data_dir, mode), transform=transform)
    except BaseException:
        dataset = ImageDataset(data_dir=data_dir, transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    output_dir = Path(output_dir) / mode
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if save:
        for images, ids, labels, img_ids in loader:
            for i, (image, label) in enumerate(zip(images, labels)):
                image_path = Path(output_dir) / label
                Path(image_path).mkdir(parents=True, exist_ok=True)
                try:
                    save_image(image, Path(image_path) / img_ids[i])
                except BaseException:
                    import pdb; pdb.set_trace()
                    continue
    return loader

def imgshow(image, title=None):
    """ This function is used to display the image. """
    # convert PIL image to numpy array
    #image = image.numpy().transpose((1, 2, 0))

    image = np.asarray(image)
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
@click.option("--is_train", is_flag=True, default=False, help="Preprocess training data")
def main(config_path, is_train):
    config = load_yaml_config(config_path)
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    batch_size = config["batch_size"]
    preprocess_images(data_dir=data_dir, output_dir=output_dir, is_train=is_train, batch_size=batch_size)


if __name__ == "__main__":
    main()


