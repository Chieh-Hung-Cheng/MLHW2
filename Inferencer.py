import torch
import os
import tqdm
import csv

class Inferencer:
    def __init__(self, config, ModelClass, save_name, test_loader):
        self.config = config
        self.model = ModelClass(5)
        self.model = self.model.to(config["device"])
        self.model.load_state_dict(torch.load(os.path.join(config["save_path"], f"model_{save_name}.ckpt")))
        self.test_loader = test_loader
        self.save_name = save_name

    def infer(self):
        self.model.eval()
        preds = []
        for x_b in tqdm.tqdm(self.test_loader):
            x_b = x_b.to(self.config["device"])
            with torch.no_grad():
                y_pred = self.model(x_b)
                y_pred = torch.argmax(y_pred, dim=1)
                preds.append(y_pred.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        self.save_pred(preds, os.path.join(self.config["output_path"], f"pred_{self.save_name}.csv"))

    def save_pred(self, preds, file):
        ''' Save predictions to specified file '''
        with open(file, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['Id', 'Class'])
            for i, p in enumerate(preds):
                writer.writerow([i, p])


if __name__ == "__main__":
    pass