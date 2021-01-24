import os
import torch
from tqdm import tqdm

class Trainer():
    def __init__(self, model, optimizer, lr_scheduler, criterion,
                 dataloaders_dict, num_epochs, device, model_save_dir="models"):

        self.optimizer = optimizer
        self.model = model.to(device)
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.dataloaders = dataloaders_dict

        self.num_epochs = num_epochs
        self.device = device
        self.best_val_loss = float("inf")
        self.best_model_weights = None
        self.model_save_dir = model_save_dir

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print("-"*10)

            for phase in ["train", "val"]:
                if phase == "train":
                    loss, acc = self.training_step(phase)
                elif phase == "val":
                    loss, acc = self.training_step(phase)
                    self.lr_scheduler.step()
                    if loss < self.best_val_loss:
                        self.best_val_loss = loss
                        self.best_model_weights = self.best_model_weights
                        if self.model_save_dir:
                            self.save_model(epoch)
                
                self.print_stats(phase, loss, acc, loss < self.best_val_loss)
        print("Done!")

    def training_step(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "val":
            self.model.eval()

        losses, accuracies = [], []

        for inputs, labels in tqdm(self.dataloaders[phase]):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.set_grad_enabled(phase=="train"):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                if phase == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                acc = torch.sum(preds == labels.data) / labels.size(0)
                
                losses.append(loss.item()*inputs.size(0))
                accuracies.append(acc)
        
        return torch.mean(losses), torch.mean(accuracies)

    def print_stats(self, phase, loss, acc, best_model_yet):
        stats = {"phase": phase, "loss": loss, "acc": acc, "best_model_yet": best_model_yet}
        for stat in stats.keys():
            print(f"{stat}: {stats[stat]}")
        print("")

    def save_model(self, epoch):
        if not os.path.isdir(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        save_path = os.path.join(self.model_save_dir, str(epoch) + "_best" + ".ckpt")
        print("Saving checkpoint to {}".format(save_path))

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_sche': self.lr_scheduler.state_dict(),
            'epoch': epoch
        }, save_path)
