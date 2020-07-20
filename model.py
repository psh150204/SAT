import torch
import torch.nn as nn
from torchvision.models import vgg19


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = vgg19(pretrained=True)
        self.net = nn.Sequential(*list(self.net.features.children())[:-1])

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), x.size(1), -1).transpose(1,2)
        return x