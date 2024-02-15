import os

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "PretrainedVGG",
    "pretrained_vgg11_bn",
    "pretrained_vgg13_bn",
    "pretrained_vgg16_bn",
    "pretrained_vgg19_bn",
]


class PretrainedVGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        super().__init__()
        self.features = features
        # CIFAR 10 (7, 7) to (1, 1)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            # nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            
        )
        self.n_features = 4096
        self.fc = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
         
        if init_weights:
            self._initialize_weights()

    def forward(self, x:Tensor, return_feature_embedding:bool=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if return_feature_embedding:
            return x 
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def pretrained_make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


pretrained_cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def get_pretrained_vgg_bn(filepath='./vgg11_bn.pt', device="cpu"):
    arch = filepath.split('/')[-1].split('.')[-2]
    if arch.split('_')[-2][-2:] == "11":
        model = PretrainedVGG(pretrained_make_layers(pretrained_cfgs["A"], batch_norm=True))
    elif arch.split('_')[-2][-2:] == "13":
        model = PretrainedVGG(pretrained_make_layers(pretrained_cfgs["B"], batch_norm=True))
    elif arch.split('_')[-2][-2:] == "16":
        model = PretrainedVGG(pretrained_make_layers(pretrained_cfgs["D"], batch_norm=True))
    elif arch.split('_')[-2][-2:] == "19":
        model = PretrainedVGG(pretrained_make_layers(pretrained_cfgs["E"], batch_norm=True))
    else:
        raise ValueError(f"Received arch={arch}. Only implemented for pretrained_vgg[16, 13, 16 or 19]_bn (batch_norm=True)")
    
    state_dict = torch.load(
        filepath, map_location=device
    )
    model.load_state_dict(state_dict)
    return model 

def _pretrained_vgg(arch, cfg, batch_norm, pretrained, progress, device, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = PretrainedVGG(pretrained_make_layers(pretrained_cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def pretrained_vgg11_bn(pretrained=False, progress=True, device="cpu", **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _pretrained_vgg("pretrained_vgg11_bn", "A", True, pretrained, progress, device, **kwargs)


def pretrained_vgg13_bn(pretrained=False, progress=True, device="cpu", **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _pretrained_vgg("pretrained_vgg13_bn", "B", True, pretrained, progress, device, **kwargs)


def pretrained_vgg16_bn(pretrained=False, progress=True, device="cpu", **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _pretrained_vgg("pretrained_vgg16_bn", "D", True, pretrained, progress, device, **kwargs)


def pretrained_vgg19_bn(pretrained=False, progress=True, device="cpu", **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _pretrained_vgg("pretrained_vgg19_bn", "E", True, pretrained, progress, device, **kwargs)