"""
MobilenetV2-style encoder network
This is based on the implementation of MobileNetV2 in pytorch 1.1 / torchvision.models
"""
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url


class MyReLU6(nn.Module):
    """ ReLU6 emulation by composition (for ONNX export) """

    def __init__(self, inplace=False):
        super(MyReLU6, self).__init__()
        self.threshold = torch.from_numpy(np.array(6., dtype='f4'))
        self.ReLU = nn.ReLU(inplace=inplace)  # pylint: disable=invalid-name

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor):
        if x.is_cuda:
            self.threshold = self.threshold.cuda(x.device)
        return -self.ReLU(self.threshold - self.ReLU(x)) + self.threshold



class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1,
                 use_custom_relu6=False):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU6, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            MyReLU6(inplace=True) if use_custom_relu6 else nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_custom_relu6=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []

        if expand_ratio != 1:
            # pw
            layers.append(
                ConvBNReLU6(inp, hidden_dim, kernel_size=1, use_custom_relu6=use_custom_relu6))

        layers.extend([
            # dw
            ConvBNReLU6(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim,
                        use_custom_relu6=use_custom_relu6),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2Encoder(nn.Module):
    """
    MobileNetV2 encoder decoder
    """

    # pylint: disable=unused-argument
    def __init__(self, pretrained=True, use_custom_relu6=False, width_mult=1.,
                 use_last_layer=True, normalize_input=False, num_layers=1):
        """
        Args:
            use_custom_relu6 (bool): Uses ReLU6, if True.
            width_mult:
        """
        super(MobileNetV2Encoder, self).__init__()
        self.use_last_layer = use_last_layer
        # imagenet statistics
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize_input = normalize_input

        # setting of inverted residual blocks
        self.inverted_residual_settings = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2]
        ]

        # building first layer
        self.width_mult = width_mult
        input_channel = int(32 * self.width_mult)


        num_ch_enc = [input_channel]

        features = [ConvBNReLU6(3, input_channel, stride=2, use_custom_relu6=use_custom_relu6)]
        # building inverted residual blocks
        for expand_ratio, num_ch_out, num_rep, stride in self.inverted_residual_settings:
            output_channel = int(num_ch_out * self.width_mult)
            for rep in range(num_rep):
                features.append(
                    InvertedResidual(inp=input_channel,
                                     oup=output_channel,
                                     stride=stride if rep == 0 else 1,
                                     expand_ratio=expand_ratio,
                                     use_custom_relu6=use_custom_relu6)
                )
                input_channel = output_channel
                if stride == 2 and rep == 0:
                    num_ch_enc.append(output_channel)

        if self.use_last_layer:
            last_channel = 1280
            features.append(ConvBNReLU6(input_channel, last_channel, kernel_size=1))
            num_ch_enc[-1] = last_channel

        self.features = nn.ModuleList(features)
        self.encoder_features = []
        self._initialize_weights()
        if pretrained:
            state_dict = load_url('http://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
                                  progress=True)
            self.load_state_dict(state_dict, strict=False)

        self.num_ch_enc = np.asarray(num_ch_enc)  # 32, 24, 32, 64, 160 or 1280

    def forward(self, input_image):
        """ forward pass of resnet """
        x = input_image
        if self.normalize_input:
            for t, m, s in zip(x, self.mean, self.std):
                t.sub(m).div(s)

        activations = [self.features[0](x)]
        self.encoder_features = [activations[0]]

        layer_idx = 1

        for expand_ratio, num_ch_out, num_rep, stride in self.inverted_residual_settings:
            for rep in range(num_rep):

                activations.append(self.features[layer_idx](activations[-1]))
                layer_idx += 1

                if stride == 2 and rep == 0:
                    self.encoder_features.append(activations[-1])

        if self.use_last_layer:
            self.encoder_features[-1] = (self.features[layer_idx](activations[-1]))

        return self.encoder_features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:  # never actually called
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):  # never actually called
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
