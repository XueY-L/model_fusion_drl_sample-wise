from functools import partial
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision.models as tmodels

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
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
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
        self.dilation = dilation
        self.groups = groups

    def forward(self, x, param, conv_module_name=None, bn_module_name=None) -> Tensor:
        identity = x

        if param == None:
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
        else:
            out = F.conv2d(input=x, weight=param[conv_module_name[0]+'.weight'])
            running_mean, running_var = out.mean([0, 2, 3]).clone().detach(), out.var([0, 2, 3], unbiased=False).clone().detach()
            out = F.batch_norm(input=out, running_mean=running_mean, running_var=running_var, weight=param[bn_module_name[0]+'.weight'], bias=param[bn_module_name[0]+'.bias'])
            out = self.relu(out)

            out = F.conv2d(input=out, weight=param[conv_module_name[1]+'.weight'], stride=self.stride, padding=self.dilation, groups=self.groups, dilation=self.dilation)
            running_mean, running_var = out.mean([0, 2, 3]).clone().detach(), out.var([0, 2, 3], unbiased=False).clone().detach()
            out = F.batch_norm(input=out, running_mean=running_mean, running_var=running_var, weight=param[bn_module_name[1]+'.weight'], bias=param[bn_module_name[1]+'.bias'])
            out = self.relu(out)

            out = F.conv2d(input=out, weight=param[conv_module_name[2]+'.weight'])
            running_mean, running_var = out.mean([0, 2, 3]).clone().detach(), out.var([0, 2, 3], unbiased=False).clone().detach()
            out = F.batch_norm(input=out, running_mean=running_mean, running_var=running_var, weight=param[bn_module_name[2]+'.weight'], bias=param[bn_module_name[2]+'.bias'])
            
            if self.downsample is not None:
                identity = F.conv2d(input=x, weight=param[conv_module_name[3]+'.weight'], stride=self.downsample[0].stride, padding=self.downsample[0].padding)
                running_mean, running_var = identity.mean([0, 2, 3]).clone().detach(), identity.var([0, 2, 3], unbiased=False).clone().detach()
                identity = F.batch_norm(input=identity, running_mean=running_mean, running_var=running_var, weight=param[bn_module_name[3]+'.weight'], bias=param[bn_module_name[3]+'.bias'])
            out += identity
            out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        width_factor = 1,  # our method
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * width_factor)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(64 * width_factor), layers[0])
        self.layer2 = self._make_layer(block, int(128 * width_factor), layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(256 * width_factor), layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(512 * width_factor), layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * width_factor * block.expansion), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(
        self,
        block,
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

    def _forward_impl(self, inp) -> Tensor:
        '''
        fuse all parameters
        '''
        x = inp[0]
        param = inp[1]
        # x = self.conv1(x)
        x = F.conv2d(input=x, weight=param['conv1.weight'], stride=self.conv1.stride, padding=self.conv1.padding)

        
        # 手动算均值方差？
        running_mean = x.mean([0, 2, 3]).clone().detach()
        running_var = x.var([0, 2, 3], unbiased=False).clone().detach()
        x = F.batch_norm(
            input=x, 
            running_mean=running_mean, 
            running_var=running_var,
            weight=param['bn1.weight'], 
            bias=param['bn1.bias']
        )

        x = self.relu(x)
        x = self.maxpool(x)

        # x = self.layer1(x)
        x = self.layer1[0](
            x, 
            param, 
            conv_module_name=['layer1.0.conv1', 'layer1.0.conv2', 'layer1.0.conv3', 'layer1.0.downsample.0'],
            bn_module_name=['layer1.0.bn1', 'layer1.0.bn2', 'layer1.0.bn3', 'layer1.0.downsample.1'],
        )
        x = self.layer1[1](
            x, 
            param, 
            conv_module_name=['layer1.1.conv1', 'layer1.1.conv2', 'layer1.1.conv3',],
            bn_module_name=['layer1.1.bn1', 'layer1.1.bn2', 'layer1.1.bn3',],
        )
        x = self.layer1[2](
            x, 
            param, 
            conv_module_name=['layer1.2.conv1', 'layer1.2.conv2', 'layer1.2.conv3',],
            bn_module_name=['layer1.2.bn1', 'layer1.2.bn2', 'layer1.2.bn3',],
        )

        # x = self.layer2(x)
        x = self.layer2[0](
            x, 
            param, 
            conv_module_name=['layer2.0.conv1', 'layer2.0.conv2', 'layer2.0.conv3', 'layer2.0.downsample.0'],
            bn_module_name=['layer2.0.bn1', 'layer2.0.bn2', 'layer2.0.bn3', 'layer2.0.downsample.1'],
        )
        x = self.layer2[1](
            x, 
            param, 
            conv_module_name=['layer2.1.conv1', 'layer2.1.conv2', 'layer2.1.conv3',],
            bn_module_name=['layer2.1.bn1', 'layer2.1.bn2', 'layer2.1.bn3',],
        )
        x = self.layer2[2](
            x, 
            param, 
            conv_module_name=['layer2.2.conv1', 'layer2.2.conv2', 'layer2.2.conv3',],
            bn_module_name=['layer2.2.bn1', 'layer2.2.bn2', 'layer2.2.bn3',],
        )
        x = self.layer2[3](
            x, 
            param, 
            conv_module_name=['layer2.3.conv1', 'layer2.3.conv2', 'layer2.3.conv3',],
            bn_module_name=['layer2.3.bn1', 'layer2.3.bn2', 'layer2.3.bn3',],
        )

        # x = self.layer3(x)
        x = self.layer3[0](
            x, 
            param, 
            conv_module_name=['layer3.0.conv1', 'layer3.0.conv2', 'layer3.0.conv3', 'layer3.0.downsample.0'],
            bn_module_name=['layer3.0.bn1', 'layer3.0.bn2', 'layer3.0.bn3', 'layer3.0.downsample.1'],
        )
        x = self.layer3[1](
            x, 
            param, 
            conv_module_name=['layer3.1.conv1', 'layer3.1.conv2', 'layer3.1.conv3'],
            bn_module_name=['layer3.1.bn1', 'layer3.1.bn2', 'layer3.1.bn3'],
        )
        x = self.layer3[2](
            x, 
            param, 
            conv_module_name=['layer3.2.conv1', 'layer3.2.conv2', 'layer3.2.conv3'],
            bn_module_name=['layer3.2.bn1', 'layer3.2.bn2', 'layer3.2.bn3'],
        )
        x = self.layer3[3](
            x, 
            param, 
            conv_module_name=['layer3.3.conv1', 'layer3.3.conv2', 'layer3.3.conv3'],
            bn_module_name=['layer3.3.bn1', 'layer3.3.bn2', 'layer3.3.bn3'],
        )
        x = self.layer3[4](
            x, 
            param, 
            conv_module_name=['layer3.4.conv1', 'layer3.4.conv2', 'layer3.4.conv3'],
            bn_module_name=['layer3.4.bn1', 'layer3.4.bn2', 'layer3.4.bn3'],
        )
        x = self.layer3[5](
            x, 
            param, 
            conv_module_name=['layer3.5.conv1', 'layer3.5.conv2', 'layer3.5.conv3'],
            bn_module_name=['layer3.5.bn1', 'layer3.5.bn2', 'layer3.5.bn3'],
        )

        # x = self.layer4(x)
        x = self.layer4[0](
            x, 
            param, 
            conv_module_name=['layer4.0.conv1', 'layer4.0.conv2', 'layer4.0.conv3', 'layer4.0.downsample.0'],
            bn_module_name=['layer4.0.bn1', 'layer4.0.bn2', 'layer4.0.bn3', 'layer4.0.downsample.1'],
        )
        x = self.layer4[1](
            x, 
            param, 
            conv_module_name=['layer4.1.conv1', 'layer4.1.conv2', 'layer4.1.conv3'],
            bn_module_name=['layer4.1.bn1', 'layer4.1.bn2', 'layer4.1.bn3'],
        )
        x = self.layer4[2](
            x, 
            param, 
            conv_module_name=['layer4.2.conv1', 'layer4.2.conv2', 'layer4.2.conv3'],
            bn_module_name=['layer4.2.bn1', 'layer4.2.bn2', 'layer4.2.bn3'],
        )

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        x = F.linear(input=x, weight=param['fc.weight'], bias=param['fc.bias'])

        return x

    def forward(self, inp) -> Tensor:
        return self._forward_impl(inp)


def _resnet(
    block,
    layers: List[int],
    progress: bool,
    **kwargs: Any,
) -> ResNet:

    model = ResNet(block, layers, **kwargs)

    return model


# def resnet18(*, progress: bool = True, **kwargs: Any) -> ResNet:

#     return _resnet(BasicBlock, [2, 2, 2, 2], progress, **kwargs)

# def resnet26(*, progress: bool = True, **kwargs: Any) -> ResNet:

#     return _resnet(BasicBlock, [3, 3, 3, 3], progress, **kwargs)

# def resnet34(*, progress: bool = True, **kwargs: Any) -> ResNet:

#     return _resnet(BasicBlock, [4, 4, 4, 4], progress, **kwargs)

def resnet50(*, progress: bool = True, **kwargs: Any) -> ResNet:

    return _resnet(Bottleneck, [3, 4, 6, 3], progress, **kwargs)


if __name__ == '__main__':
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    data = torch.randn(200,3,224,224)

    template_model = tmodels.resnet50(pretrained=True)
    template_model.train()
    model = resnet50()
    model.eval()

    # 钩子函数，用于截取指定层的输出
    def hook_fn(module, input, output):
        global inp, outp
        inp = torch.clone(input[0].detach())
        outp = torch.clone(output.detach())
    # 注册钩子
    target_layer = template_model.fc
    hook = target_layer.register_forward_hook(hook_fn)
    rst = template_model(data)
    hook.remove()
    _, predicted = torch.max(rst.data, 1)

    ttt = model(data, template_model.state_dict())
    _, predicted_f = torch.max(ttt.data, 1)
    print(predicted[152:165])
    print(predicted_f[152:165])
