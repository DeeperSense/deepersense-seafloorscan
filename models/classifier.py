import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, in_features, num_classes):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Conv2d(in_features, num_classes, kernel_size=(1, 1), bias=False) 

    def forward(self, x):
        x = torch.mean(x, [2, 3], keepdim=True)
        x = self.classifier(x)
        x = x.view(-1,self.num_classes)
        return x

    def calculate_cam(self, x):
        # input: list of 2 tensors of shape (B,C,h,w)
        # output: tensor of shape (B,K,h,w)
        with torch.set_grad_enabled(False):
            weights = torch.zeros_like(self.classifier.weight)
            with torch.no_grad():
                weights.set_(self.classifier.weight.detach())
            x = [F.relu(F.conv2d(x_, weight=weights)) for x_ in x]
            x = x[0] + x[1].flip(-1)
            return x
