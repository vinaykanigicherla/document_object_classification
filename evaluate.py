import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

from models import make_model
from dataset import DOCDataset

def test(model_path):
    data_dir = "data/test"
    num_classes = 3
    batch_size = 0
    device = torch.device("cuda:0" if torch.cuda.device.is_available() else "cpu")


    checkpoint = torch.load(model_path)
    model_ft = make_model("squeezenet", num_classes)
    model_ft.load_state_dict(checkpoint["model_state_dict"])
    
    dataloader = DataLoader(dataset=DOCDataset(data_dir, test), batch_size=batch_size)

    correct, total = 0
    for img, label in tqdm(dataloader):
        with torch.no_grad():
            pass
