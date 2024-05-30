import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MobilnetWEdittedClassifier(nn.Module):
    def __init__(self, num_classes=64, num_plant_types=15):
        super(MobilnetWEdittedClassifier, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=False),
            nn.Linear(in_features=1280 + num_plant_types, out_features=num_classes, bias=True)
        )

    def forward(self, x, plant_type):
        x = self.mobilenet.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = torch.cat((x, plant_type), dim=1)
        x = self.mobilenet.classifier(x)
        return x
            