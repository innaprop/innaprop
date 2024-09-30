import torch
import torch.nn as nn
import sys
sys.path.append('/code/cnn-inna/')

class LeNet5(nn.Module):
    def __init__(self, num_classes=10, batch_norm=True):
        super(LeNet5, self).__init__()
        self.batch_norm = batch_norm

        self.features = self._make_layers()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=16*4*4, out_features=120),
            nn.ReLU(True),
            nn.BatchNorm1d(120) if self.batch_norm else nn.Identity(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(True),
            nn.BatchNorm1d(84) if self.batch_norm else nn.Identity(),
            nn.Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self):
        layers = [
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(6) if self.batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16) if self.batch_norm else nn.Identity(),
        ]
        return nn.Sequential(*layers)


def test():
    net = LeNet5(maxpool_fn=lambda: MaxPool2dAlpha(0))
    x = torch.randn(2,1,28,28)
    y = net(x)
    print(y.size())
    print(net)