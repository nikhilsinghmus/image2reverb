import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, calculate_gain


class PixelWiseNormLayer(nn.Module):
    """PixelNorm layer. Implementation is from https://github.com/shanexn/pytorch-pggan."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x/torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class MiniBatchAverageLayer(nn.Module):
    """Minibatch stat concatenation layer. Implementation is from https://github.com/shanexn/pytorch-pggan."""
    def __init__(self, offset=1e-8):
        super().__init__()
        self.offset = offset

    def forward(self, x):
        stddev = torch.sqrt(torch.mean((x - torch.mean(x, dim=0, keepdim=True))**2, dim=0, keepdim=True) + self.offset)
        inject_shape = list(x.size())[:]
        inject_shape[1] = 1
        inject = torch.mean(stddev, dim=1, keepdim=True)
        inject = inject.expand(inject_shape)
        return torch.cat((x, inject), dim=1)


class EqualizedLearningRateLayer(nn.Module):
    """Applies equalized learning rate to the preceding layer. Implementation is from https://github.com/shanexn/pytorch-pggan."""
    def __init__(self, layer):
        super().__init__()
        self.layer_ = layer

        kaiming_normal_(self.layer_.weight, a=calculate_gain("conv2d"))
        self.layer_norm_constant_ = (torch.mean(self.layer_.weight.data ** 2)) ** 0.5
        self.layer_.weight.data.copy_(self.layer_.weight.data / self.layer_norm_constant_)

        self.bias_ = self.layer_.bias if self.layer_.bias else None
        self.layer_.bias = None

    def forward(self, x):
        self.layer_norm_constant_ = self.layer_norm_constant_.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor)
        x = self.layer_norm_constant_ * x
        if self.bias_ is not None:
            x += self.bias.view(1, self.bias.size()[0], 1, 1)
        return x


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")
