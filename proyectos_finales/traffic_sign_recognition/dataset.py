from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

Signal_map = {
    0: "20 Vel",
    1: "30 Vel",
    2: "50 Vel",
    3: "60 Vel",
    4: "70 Vel",
    5: "80 Vel",
    6: "Not 80 Vel",
    7: "100 Vel",
    8: "120 Vel",
    9: "Car Overtake",
    10: "Truck Overtake",
    11: "Priority Crossroad",
    12:  "Hazard Ahead",
    13: "Yield",
    14: "Stop",
    15: "Driving Safe",
    16: "No Truck",
    17: "No Entry",
    18: "Caution",
    19: "No Turn Left",
    20: "No Turn Right",
    21: "S Curve",
    22: "Bumps Ahead",
    23: "Slipery Road",
    24: "Road Closing",
    25: "Work Ahead",
    26: "Traffic light Ahead",
    27: "Crossing Ahead",
    28: "School Crossing Ahead",
    29: "Bycicle Crossing Ahead",
    30: "Snow Ahead",
    31: "Animals Warning",
    32: "Forbidden",
    33: "Turn RIght",
    34: "Turn Left",
    35: "Go Ahead",
    36: "Ahead or Right",
    37: "Ahead or Left",
    38: "Keep Right",
    39: "Keep Left",
    40: "Roundabout",
    41: "No Car Overtake",
    42: "No Truck Overtake"
}

def get_dataloaders(batch_size):
    """Returns train and validation dataloaders for the traffic sign recognition dataset"""
    file_path = Path(__file__).parent.absolute()
    root_path = file_path / "data/crop_dataset/crop_dataset/"

    # https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html    
    dataset = ImageFolder(root=root_path,
                          transform=get_transform())

    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    return train_dataloader, val_dataloader


def get_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
        ]
    )


def visualize_data():
    train_dataloader, val_dataloader = get_dataloaders(batch_size=64)

    # Visualize some training images
    for data, target in train_dataloader:
        img_grid = make_grid(data)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.show()
        print(data.shape)
        print(target.shape)
        break

    # Visualize some validation images with labels
    for data, target in val_dataloader:
        plt.figure(figsize=(8, 8))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.axis("off")
            plt.imshow(data[i].permute(1, 2, 0))
            plt.title(target[i].item())
        plt.show()
        break


def main():
    visualize_data()


if __name__ == "__main__":
    main()