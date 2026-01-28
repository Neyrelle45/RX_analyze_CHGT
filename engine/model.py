import torch.nn as nn
import torch.nn.functional as F

class UNetLite(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(16,32,3,padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.dec1 = nn.Sequential(nn.Conv2d(32,16,3,padding=1), nn.ReLU())
        self.out  = nn.Conv2d(16,n_classes,1)

    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = F.interpolate(e2, scale_factor=2)
        d1 = self.dec1(d1)
        return self.out(d1)
