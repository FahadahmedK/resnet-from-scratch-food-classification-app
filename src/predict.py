import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from utils import load_yaml_config
from architecture import ResNet


def predict(model, image):
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = transform_test(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()


@click.command()
@click.option("--image_path", type=click.Path(exists=True))
def main(image_path):
    config = load_yaml_config("configs/prediction.yaml")
    model_path = config["model_path"]

    image = Image.open(image_path)
    model = ResNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    prediction = predict(model, image)

    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
