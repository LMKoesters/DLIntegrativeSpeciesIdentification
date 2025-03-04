import torch
from torch import nn
from torchvision import models


class BarcodeJPGImg:
    def __init__(self, num_classes, classifier_mode, w_classifier=True):
        self.num_classes = num_classes
        self.model, self.num_ftrs = self.initialize_resnet()
        self.w_classifier = w_classifier
        self.model.fc = self.get_classifier(classifier_mode)

    def count_model_params(self) -> tuple[None, int, int, None, int]:
        classifier = sum(map(torch.numel, self.model.fc.parameters()))
        total = sum(map(torch.numel, self.model.parameters()))
        ftrs = total - classifier
        
        return None, ftrs, classifier, None, total

    # set all parameters to trainable (will be overwritten when fused)
    def set_parameter_requires_grad(self, model):
        for name, param in model.named_parameters():
            param.requires_grad = True

    # get resnet and set parameters to  trainable
    def initialize_resnet(self):
        model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.fc.in_features

        return model_ft, num_ftrs

    # set up classifier according to fusion strategy
    def get_classifier(self, classifier_mode):
        if self.w_classifier:
            if classifier_mode == 'slf' or classifier_mode == 'dense_late' or classifier_mode == 'sep':
                num_neurons = 768
            else:
                num_neurons = 1024
            
            classifier = nn.Sequential(
                nn.Linear(self.num_ftrs, num_neurons),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(num_neurons, self.num_classes),
                nn.Softmax(dim=1)
            )
        else:
           classifier = torch.nn.Identity() 
        
        return classifier
