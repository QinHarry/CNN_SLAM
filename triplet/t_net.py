import torch.nn as nn
import torch

class TNet(nn.Module):
    """TFeat model definition
    """
    def __init__(self):
        super(TNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7),
            #nn.BatchNorm2d(32),
            nn.Tanh(),
            #nn.FractionalMaxPool2d(kernel_size=2, output_ratio=0.5),
            nn.AdaptiveMaxPool2d(output_size=(13, 13)),
            nn.Conv2d(32, 64, kernel_size=6),
            #nn.BatchNorm2d(64),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x