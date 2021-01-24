import os
from cv2 import cv2
from torch.utils.data import Dataset

class DOCDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        assert split in ["train", "val", "test"]
        self.data_dir = data_dir
        self.split = split
        self.imgs = os.listdir(os.path.join(self.data_dir, self.split))
        self.transform = transform
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = cv2.imread(img_path)
        img = self.transform(img)
        label = int(img_path.split("_")[0])

        return (img, label)