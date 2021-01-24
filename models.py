import torch
import torch.nn as nn
from torchvision import models

def make_model(name, num_classes):
    if name == "squeezenet":
        model = models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
        return model