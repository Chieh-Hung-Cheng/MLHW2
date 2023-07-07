import torch
import torch.nn as nn

class SoundNetwork(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.F = 2*window + 1 # 11
        # input is (2*window, 39): (F=11, 39)
        # forward: change to (39, F=11)
        f0 = self.F - 3 + 1
        f1 = f0 - 3 + 1
        self.net = nn.Sequential(
            nn.Conv1d(39, 30, 3),
            nn.ReLU(),
            nn.Conv1d(30, 15, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(f1 * 15, 80),
            nn.ReLU(),
            nn.Linear(80, 60),
            nn.ReLU(),
            nn.Linear(60, 41),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
        self.conv_0 = nn.Conv1d(39, 30, 3) # (30, F-3+1(9))
        self.conv_1 = nn.Conv1d(30, 15, 3) # (15, F-3+1-3+1(7))
        # flatten (105)
        self.linear_0 = nn.Linear(f1*15, 80)
        self.lienar_1 = nn.Linear(80, 60)
        self.linear_2 = nn.Linear(60, 41)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.permute(0, 2 ,1)
        x = self.net(x)
        return x.squeeze()




if __name__ == "__main__":
    sn = SoundNetwork(5).to('cuda')
    x_rand = torch.rand(64, 11, 39)
    x_rand = x_rand.to('cuda')
    y = sn(x_rand)
    x = 1