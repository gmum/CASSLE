from typing import Any, Type, Union, List, Dict

import torch
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls

from models import reset_parameters


class ResnetOutBlocks(ResNet):
    def _forward_impl(self, x: Tensor) -> Dict[str, Tensor]:
        # See note [TorchScript super()]
        in_x=x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        conv1_out = x = self.maxpool(x)

        l1 = x = self.layer1(x)
        l2 = x = self.layer2(x)
        l3 = x = self.layer3(x)
        l4 = x = self.layer4(x)

        x = self.avgpool(x)
        backbone_out = x = torch.flatten(x, 1)
        x = self.fc(x)

        return dict(
            input=in_x,
            conv1=conv1_out,
            l1=l1,
            l2=l2,
            l3=l3,
            l4=l4,
            backbone_out = backbone_out,
            out=x
        )

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResnetOutBlocks:
    model = ResnetOutBlocks(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResnetOutBlocks:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResnetOutBlocks:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def load_backbone_out_blocks(args):
    name = args.model

    if name == "resnet18":
        backbone = resnet18(zero_init_residual=True)
    elif name == "resnet50":
        backbone = resnet50(zero_init_residual=True)
    else:
        raise NotImplementedError(name)

    if name.startswith('cifar_'):
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
    args.num_backbone_features = backbone.fc.weight.shape[1]
    backbone.fc = nn.Identity()
    reset_parameters(backbone)
    return backbone