from tqdm import tqdm
import SoundNetwork
import torch
import os
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, valid_loader, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config

    def train(self, epochs):
        device = self.config["device"]
        best_loss = None
        best_accuracy = None
        early_stop_count = 0
        writer = SummaryWriter()

        epochs_pbar = tqdm(range(epochs))
        for epoch in epochs_pbar:
            # train
            self.model.train()
            loss_record = []
            for x_b, y_b in self.train_loader:
                self.optimizer.zero_grad()
                x_b = x_b.to(device) # (N,11,39)
                y_b = y_b.to(device)
                y_pred = self.model(x_b)

                loss = self.criterion(y_pred, y_b)
                loss.backward()
                self.optimizer.step()

                loss_record.append(loss.detach().item())

            mean_train_loss = sum(loss_record) / len(loss_record)
            writer.add_scalar("TrainLoss", mean_train_loss, epoch)

            # valid
            self.model.eval()
            loss_record = []
            correct_count = 0
            item_count = 0
            for x_b, y_b in self.valid_loader:
                x_b = x_b.to(device)
                y_b = y_b.to(device)
                y_pred = self.model(x_b)

                loss = self.criterion(y_pred, y_b)
                loss_record.append(loss.detach().item())

                label_pred = torch.argmax(y_pred, dim=1)
                label_truth = torch.argmax(y_b, dim=1)
                correct_count += torch.sum(label_pred == label_truth)
                item_count += len(y_b)

            mean_valid_loss = sum(loss_record) / len(loss_record)
            writer.add_scalar("ValidLoss", mean_valid_loss, epoch)

            # Update tqdm
            epochs_pbar.set_description(f"{epoch}/{epochs}")
            epochs_pbar.set_postfix({"train_loss": f"{mean_train_loss:.4f}",
                                     "valid_loss": f"{mean_valid_loss:.4f}",
                                     "early_countdown": f"{self.config['early_stop']-early_stop_count}",
                                     "accuracy": f"{correct_count/item_count:.2%}"})

            # Saving and Early Stopping
            if (best_loss is None) or (mean_valid_loss<best_loss):
                best_loss = mean_valid_loss
                best_accuracy = f"{correct_count/item_count:.2%}"
                self.save_current_model(best_loss)
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= self.config["early_stop"]:
                    print(f"Early Stopped at {epoch=}")
                    break

        with open(os.path.join(self.config["save_path"], "model_log"), "a") as file:
            file.write(f"{self.config['time_string']}, {best_loss=}, {best_accuracy=}")


    def save_current_model(self, loss):
        torch.save(self.model.state_dict(),
                   os.path.join(self.config["save_path"], f"model_{self.config['time_string']}.ckpt"))