import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms
from .layers import PixelWiseNormLayer, MiniBatchAverageLayer, EqualizedLearningRateLayer


class Encoder(nn.Module):
    """Load encoder from pre-trained ResNet50 (places365 CNNs) model. Link: http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"""
    def __init__(self, model_weights, device="cuda"):
        super().__init__()
        self.model_weights = model_weights
        self.model = models.resnet50(num_classes=365)
        c = torch.load(model_weights, map_location=device)
        state_dict = {k.replace("module.", ""): v for k, v in c["state_dict"].items()}
        self.model.load_state_dict(state_dict)
        self.model.to(torch.device(device))
        self.model.train()

    def forward(self, x):
        return self.model.forward(x).unsqueeze(-1).unsqueeze(-1)
    
    def recursion_change_bn(self, module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = 1
        else:
            for i, (name, module1) in enumerate(module._modules.items()):
                module1 = self.recursion_change_bn(module1)
        return module


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
            z = torch.cat((l.squeeze(), torch.zeros(s).cuda()), -1).reshape(d.shape[0], -1, 2, 4)
        else:
            s = list(l.squeeze().shape)
            s[-1] = 512 - s[-1]
            z = torch.cat((l.squeeze(), torch.zeros(s).cuda()), -1).reshape(d.shape[0], -1, 8, 8)
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
