import torch
import torch.nn as nn
from torchvision import models
from torch import Tensor


class CovidModel(nn.Module):
    """
    covid classifier model based on resnet
    """

    def __init__(self, pretrained: bool = True):
        """

        :param pretrained: if true it used pretrained resnet
        """
        super(CovidModel, self).__init__()
        # define resnet model then remove classifier layer
        resnet_model = models.resnet18(pretrained=pretrained)
        resnet_modules = list(resnet_model.children())  # gets all modules
        # everything is set to create out own model
        # using nn.Sequential for ease of use
        self.model = nn.Sequential(*resnet_modules[:-1])  # all layers but last
        # classifier layer
        self.classifier = nn.Linear(resnet_modules[-1].in_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        """

        :param x: input image
        :return: probability of having Covid(1 is positive)
        """

        batch_size = x.size()[0]
        x = self.model(x)
        # flatten output
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        # using sigmoid activation
        x = torch.sigmoid(x).squeeze(1)
        return x
