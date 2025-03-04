import torch
from torch import nn, Tensor
from typing import Callable, Optional, Type, Union


# the contents of this file were mostly copied
# from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# with minor adjustments for compatibility and a custom classifier for BarcodeJPGBar

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BarcodeJPGBar(nn.Module):
    def __init__(self, num_classes, classifier_mode, init_weights=True, w_classifier=True):
        super(BarcodeJPGBar, self).__init__()
        self.block = Bottleneck
        self.inplanes = 64
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.features = self.make_resnet(self.block)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.w_classifier = w_classifier
        self.num_classes = num_classes
        self.w_classifier = w_classifier
        self.fc = self.get_classifier(classifier_mode)

        if init_weights:
            self._initialize_weights()

        self.set_parameter_requires_grad()

    # set all parameters to trainable (will be overwritten when fused)
    def set_parameter_requires_grad(self):
        for m in self.modules():
            for name, param in m.named_parameters():
                param.requires_grad = True
        
    def count_model_params(self):
        ftrs = sum(map(torch.numel, self.features.parameters()))
        total = ftrs
        if self.w_classifier:
            classifier = sum(map(torch.numel, self.fc.parameters()))
            total += classifier
        else:
            classifier = 0
            
        return ftrs, None, classifier, None, total

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.w_classifier:
            x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_resnet(self, block):
        layers = [3, 4, 6, 3]

        model_layers = [nn.Conv2d(4, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)]
        model_layers += [self._norm_layer(self.inplanes)]
        model_layers += [nn.ReLU(inplace=True)]
        model_layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        model_layers += [self._make_layer(block, 64, layers[0])]
        model_layers += [self._make_layer(block, 128, layers[1], stride=2)]
        model_layers += [self._make_layer(block, 256, layers[2], stride=2)]
        model_layers += [self._make_layer(block, 512, layers[3], stride=2)]

        return nn.Sequential(*model_layers)

    def _make_layer(
            self,
            block: Type[Union[Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    # set up classifying layers for different fusion methods
    def get_classifier(self, classifier_mode):
        if self.w_classifier:
            if classifier_mode == 'slf' or classifier_mode == 'dense_late' or classifier_mode == 'sep':
                num_neurons = 768
            else:
                num_neurons = 1024
                
            classifier = nn.Sequential(
                    nn.Linear(512 * self.block.expansion, num_neurons),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(num_neurons, self.num_classes),
                    nn.Softmax(dim=1)
                )
        else:
            classifier = torch.nn.Identity()
        
        return classifier


