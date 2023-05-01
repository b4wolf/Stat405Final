import torch.nn as nn
import torch

class ModifiedNN(nn.Module):
    def __init__(self):
        super(ModifiedNN, self).__init__()

    def forward(self, x, metadata):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, metadata), dim=1)
        x = self.fc(x)
        return x
    
class ModifiedResNet(ModifiedNN):
    def __init__(self, resnet, transform, metadata_size):
        super(ModifiedResNet, self).__init__()
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features + metadata_size, 7)
        self.transform = transform


class ModifiedSwinTransformer(ModifiedNN):
    def __init__(self, swin_transformer, transform, metadata_size):
        super(ModifiedSwinTransformer, self).__init__()
        self.model = nn.Sequential(*list(swin_transformer.children())[:-1])
        self.fc = nn.Linear(swin_transformer.head.in_features + metadata_size, 7)
        self.transform = transform

class ModifiedConvNext(ModifiedNN):
    def __init__(self, convnext, transform, metadata_size):
        super(ModifiedConvNext, self).__init__()
        self.model = nn.Sequential(*list(convnext.children())[:-1])
        in_features = convnext.classifier[-1].in_features + metadata_size
        self.fc = nn.Linear(in_features, 7)
        self.transform = transform
