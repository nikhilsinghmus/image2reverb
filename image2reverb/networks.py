import os
import numpy
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from .layers import PixelWiseNormLayer, MiniBatchAverageLayer, EqualizedLearningRateLayer, Conv3x3, ConvBlock, upsample


class Encoder(nn.Module):
    """Load encoder from pre-trained ResNet50 (places365 CNNs) model. Link: http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"""
    def __init__(self, model_weights, depth_model, constant_depth=None, device="cuda", train_enc=True):
        super().__init__()
        self.device = device
        self._constant_depth = constant_depth
        self.model = models.resnet50(num_classes=365)

        if model_weights:
            c = torch.load(model_weights, map_location=self.device)
            state_dict = {k.replace("module.", ""): v for k, v in c["state_dict"].items()}
            self.model.load_state_dict(state_dict)
        
        self._has_depth = False
        if depth_model:
            f = self.model.conv1.weight
            self.model.conv1.weight = torch.nn.Parameter(torch.cat((f, torch.randn(64, 1, 7, 7)), 1))
            self.model.to(self.device)

            encoder_path = os.path.join(depth_model, "encoder.pth")
            depth_decoder_path = os.path.join(depth_model, "depth.pth")
            self.depth_encoder = ResnetEncoder(18, False)
            loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

            self.feed_height = loaded_dict_enc["height"]
            self.feed_width = loaded_dict_enc["width"]
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.depth_encoder.state_dict()}
            self.depth_encoder.load_state_dict(filtered_dict_enc)
            self.depth_encoder.to(self.device)
            self.depth_encoder.eval()

            self.depth_decoder = DepthDecoder(num_ch_enc=self.depth_encoder.num_ch_enc, scales=range(4))
            loaded_dict = torch.load(depth_decoder_path, map_location=self.device)
            self.depth_decoder.load_state_dict(loaded_dict, strict=False)
            self.depth_decoder.to(self.device)
            self.depth_decoder.eval()

            self._has_depth = True
        
        if train_enc:
            self.model.train()

    def forward(self, x):
        if self._has_depth:
            d = torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self._constant_depth, device=x.device) if self._constant_depth is not None else list(self.depth_decoder(self.depth_encoder(x)).values())[-1]
            x = torch.cat((x, d), 1)
        return self.model.forward(x).unsqueeze(-1).unsqueeze(-1), x


class Generator(nn.Module):
    """Build non-progressive variant of GANSynth generator."""
    def __init__(self, latent_size=512, mel_spec=False): # Encoder output should contain 2048 values
        super().__init__()
        self.latent_size = latent_size
        self._mel_spec = mel_spec
        self.build_model()

    def forward(self, x):
        return self.model(x)

    def build_model(self):
        model = []
        # Input block
        if self._mel_spec:
            model.append(nn.Conv2d(self.latent_size, 256, kernel_size=(4, 2), stride=1, padding=2, bias=False))
        else:
            model.append(nn.Conv2d(self.latent_size, 256, kernel_size=8, stride=1, padding=7, bias=False)) # Modified to k=8, p=7 for our image dimensions (i.e. 512x512)
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())

        model.append(nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.Tanh())
        self.model = nn.Sequential(*model)


class Discriminator(nn.Module):
    def __init__(self, label_size=365, mel_spec=False):
        super().__init__()
        self._label_size = 365
        self._mel_spec = mel_spec
        self.build_model()

    def forward(self, x, l):
        d = self.model(x)
        if self._mel_spec:
            s = list(l.squeeze().shape)
            s[-1] = 19
            z = torch.cat((l.squeeze(), torch.zeros(s).type_as(x)), -1).reshape(d.shape[0], -1, 2, 4)
        else:
            s = list(l.squeeze().shape)
            s[-1] = 512 - s[-1]
            z = torch.cat((l.squeeze(), torch.zeros(s).type_as(x)), -1).reshape(d.shape[0], -1, 8, 8)
        k = torch.cat((d, z), 1)
        return self.output(k)

    def build_model(self):
        model = []
        model.append(nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))

        model.append(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))

        model.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))

        model.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))

        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))

        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))

        model.append(MiniBatchAverageLayer())
        model.append(nn.Conv2d(257, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))

        output = [] # After the label concatenation
        if self._mel_spec:
            output.append(nn.Conv2d(304, 256, kernel_size=1, stride=1, padding=0, bias=False))
        else:
            output.append(nn.Conv2d(264, 256, kernel_size=1, stride=1, padding=0, bias=False))
            
        output.append(nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=False))

        # model.append(nn.Sigmoid()) # Output probability (in [0, 1])
        self.model = nn.Sequential(*model)
        self.output = nn.Sequential(*output)


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = numpy.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features



class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = "nearest"
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = numpy.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            # self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            setattr(self, "upconv_{}_0".format(i), ConvBlock(num_ch_in, num_ch_out))

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            # self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            setattr(self, "upconv_{}_1".format(i), ConvBlock(num_ch_in, num_ch_out))

        for s in self.scales:
            # self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            setattr(self, "disp_{}".format(s), Conv3x3(self.num_ch_dec[s], self.num_output_channels))

        self.decoder = nn.ModuleList(
            [x for y in [[getattr(self, "upconv_{}_0".format(i)), getattr(self, "upconv_{}_1".format(i))] for i in range(4, -1, -1)] for x in y] +
            [getattr(self, "disp_{}".format(s)) for s in self.scales]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            # x = self.convs[("upconv", i, 0)](x)
            x = getattr(self, "upconv_{}_0".format(i))(x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            # x = self.convs[("upconv", i, 1)](x)
            x = getattr(self, "upconv_{}_1".format(i))(x)
            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(getattr(self, "disp_{}".format(i))(x))
                # setattr(self, "outputs_disp_{}".format(i), self.sigmoid(getattr(self, "disp_{}".format(i))(x)))

        return outputs
