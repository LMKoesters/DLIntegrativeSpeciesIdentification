import torch
from torch import nn
from torchvision import models
from typing import Union

from model_bar_resnet import BarcodeJPGBar
from model_bar_resnet_sequential_embedding import SequentialEmbeddingBarcodeJPGBar


class BarcodeJPGFuse(nn.Module):
    def __init__(self, job_id: str, barcode_model: Union[BarcodeJPGBar, SequentialEmbeddingBarcodeJPGBar],
                 image_model: models.resnet50, num_classes: int, classifier_mode: str, slf_mode: str = None,
                 trainable: bool = False, input_ftrs: int = None):
        super(BarcodeJPGFuse, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.job_id = job_id
        self.barcode_model = barcode_model
        self.image_model = image_model
        self.num_classes = num_classes
        self.trainable = trainable
        self.slf_mode = slf_mode
        self.input_ftrs = input_ftrs
        self.fc = self.get_classifier(classifier_mode)
        self.set_parameter_requires_grad()

    # count model parameters
    def count_model_params(self) -> tuple[int, int, int, int, int]:
        bar_conv, _, bar_mlp, _, total = self.barcode_model.count_model_params()
        
        img_mlp = sum(map(torch.numel, self.image_model.fc.parameters()))
        img_total = sum(map(torch.numel, self.image_model.parameters()))
        img_conv = img_total - img_mlp
        
        shared = sum(map(torch.numel, self.fc.parameters()))
        mlp = img_mlp + bar_mlp + shared
        total = img_conv + bar_conv + mlp
        
        return bar_conv, img_conv, mlp, shared, total
        
    # set parameters to trainable according to training round and layer
    def set_parameter_requires_grad(self):
        for m in [self.barcode_model, self.image_model]:
            for name, param in m.named_parameters():
                if self.trainable:
                    param.requires_grad = True
                elif 'classifier' in name or 'fc.' in name:
                    param.requires_grad = True
                elif 'features.7.0.' in name or 'layer4.1.' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    # forward pass with fusion based on fusion strategy
    def forward(self, barcode, image):
        barcode_out = self.barcode_model(barcode)
        image_out = self.image_model(image)

        # fuse outputs
        if self.slf_mode is not None:
            if self.slf_mode == 'sum_score':
                fused_out = (barcode_out + image_out) / 2
            elif self.slf_mode == 'max_score':
                fused_out = torch.stack((barcode_out, image_out))
                fused_out = torch.max(fused_out, dim=0).values
            else:  # product_score
                fused_out = torch.mul(barcode_out, image_out)
        else:
            fused_out = torch.cat((barcode_out, image_out), dim=1)
            fused_out = self.fc(fused_out)

        # rescale output for score level fusion if needed (sum score was rescaled immediately)
        if self.slf_mode == 'max_score' or self.slf_mode == 'product_score':
            y = torch.sum(fused_out, dim=1).unsqueeze(1).expand(fused_out.shape[0], fused_out.shape[1])
            fused_out = torch.div(fused_out, y)

        return fused_out, barcode_out, image_out

    # set up classifier for fusion based on fusion strategy
    def get_classifier(self, classifier_mode):
        if classifier_mode == 'dense_late':
            b_ftrs = list(self.barcode_model.children())[-1][0].out_features
            i_ftrs = list(self.image_model.children())[-1][0].out_features
            input_ftrs = b_ftrs + i_ftrs
            classifier = nn.Sequential(
                nn.Linear(input_ftrs, self.num_classes),
                nn.Softmax(dim=1)
            )
        elif classifier_mode == 'dense_mid':
            input_ftrs = self.input_ftrs
            # number of neurons is set so that we achieve 2x separate model size
            if self.job_id == 'Coccinellidae':
                num_neurons = 784
            elif self.job_id == 'Lycaenidae':
                num_neurons = 816
            elif self.job_id == 'Poaceae':
                num_neurons = 790
            elif self.job_id == 'Asteraceae':
                num_neurons = 794
            else:
                print("Using default number of neurons.")
                num_neurons = 794

            classifier = nn.Sequential(
                nn.Linear(input_ftrs, num_neurons),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(num_neurons, self.num_classes),
                nn.Softmax(dim=1)
            )
        else:
            classifier = torch.nn.Identity()

        return classifier
