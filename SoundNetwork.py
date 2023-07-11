import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(p=0.5)
        )
    def forward(self, x):
        return self.block(x)


class SoundNetwork(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.F = 2*window + 1 # 11
        # input is (2*window, 39): (F=11, 39)
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1), # (429, )
            BasicBlock(429, 512),
            BasicBlock(512, 256),
            BasicBlock(256, 128),
            BasicBlock(128, 64),
            nn.Linear(64, 41)
        )


    def forward(self, x):
        x = self.net(x)
        return x.squeeze()




if __name__ == "__main__":
    sn = SoundNetwork(5).to('cuda')
    x_rand = torch.rand(64, 11, 39)
    x_rand = x_rand.to('cuda')
    y = sn(x_rand)
    x = 1