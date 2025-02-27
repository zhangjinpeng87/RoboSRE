import torch
import torchvision.models as models
import torch.nn as nn

class ResNet:
    def __init__(self, num_classes, pretrained=True):
        self.model = models.resnet18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def get_model(self):
        return self.model