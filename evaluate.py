import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

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
    for imgs, labels in tqdm(dataloader):
        with torch.no_grad():
            output = model_ft(imgs)
            probs, predicted = torch.nn.Softmax(output, dim=1).topk(1, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print("Model Accuracy: {.:2f}".format(correct/total))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model specified at path on test set")
    parser.add_argument("path", type=str, help="Path to model")
    args = parser.parse_args()
    test(args.path)

