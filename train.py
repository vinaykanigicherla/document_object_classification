import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os

from dataset import DOCDataset
from trainer import Trainer

def main():
    data_dir = "data"
    model_name = "squeezenet"
    input_size = 224
    num_epochs = 10
    num_classes = 3
    batch_size = 32 
    device = torch.device("cuda:0" if torch.cuda.device.is_available() else "cpu")

    composed_data_transforms = transforms.Compose(transforms.Resize(input_size), 
                                                transforms.ToTensor(), 
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225]))

    dataloaders_dict = {split: torch.utils.data.DataLoader(dataset=DOCDataset(data_dir, split, composed_data_transforms), batch_size=batch_size, shuffle=True)
                    for split in ["train", "val"]}


    model_ft = models.squeezenet1_0(pretrained=True)
    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model_ft.num_classes = num_classes

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_ft.parameters(), lr=0.001)
    step_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    trainer = Trainer(model_ft, optimizer, step_lr_scheduler, criterion, dataloaders_dict, num_epochs, device)
    trainer.train()

    return model_ft.load_state_dict(trainer.best_model_weights)

if __name__ == "__main__":
    best_model = main()


