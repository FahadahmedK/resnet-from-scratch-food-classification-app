from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.utils import load_yaml_config
from src.preprocess import ImageDataset
from src.architecture import ResNet
import click


#writer = SummaryWriter()


def train_step(model, train_loader, criterion, optimizer, device, writer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}: Training")
    for images, labels, _, _ in train_bar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)

        train_bar.set_postfix(train_loss=train_loss, train_acc=train_acc)
        import pdb; pdb.set_trace()
    return train_loss, train_acc


def validation_step(model, val_loader, criterion, device, writer, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}: Validation")
    with torch.no_grad():
        for images, labels, _,_ in val_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_loss = running_loss / len(val_loader)
            val_acc = correct / total
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch)

            val_bar.set_postfix(val_loss=val_loss, val_acc=val_acc)

    return val_loss, val_acc


def train_model(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, device, log_dir, model_path, model_name):

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_step(
            model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device, writer=writer, epoch=epoch
        )
        val_loss, val_acc = validation_step(
            model=model, val_loader=val_loader, criterion=criterion, device=device, writer=writer, epoch=epoch
        )
        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{num_epochs}: train_loss: {train_loss:.4f}, train_acc: {train_acc:.2f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}"
        )
        print(
            "------------------------------------------------------------------------------------------------------------------"
        )
    full_model_path = Path(model_path / model_name)
    torch.save(model.state_dict(), full_model_path)
    writer.close()


@click.command()
@click.option(
    "--config_path", default="configs/train.yaml", help="Path to config file."
)
def main(config_path):
    config = load_yaml_config(config_path)
    data_dir = config["data_dir"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]
    num_classes = config["num_classes"]
    model_name = config["model_name"]
    model_path = config["model_path"]
    log_dir = config["log_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageDataset(data_dir)
    #test_dataset = ImageDataset(data_dir/mode=test, train=False)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ResNet(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif config["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_model(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, device, log_dir, model_path, model_name)

if __name__ == "__main__":
    main()