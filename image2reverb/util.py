import os
import math
import numpy
import torch
import torch.fft
from PIL import Image


def compare_t60(a, b, sr=86):
    try:
        a = a.detach().clone().abs()
        b = b.detach().clone().abs()
        a = (a - a.min())/(a.max() - a.min())
        b = (b - b.min())/(b.max() - b.min())
        t_a = estimate_t60(a, sr)
        t_b = estimate_t60(b, sr)
        return abs((t_b - t_a)/t_a) * 100
    except Exception as error:
        return 100


def estimate_t60(audio, sr):
    fs = float(sr)
    audio = audio.detach().clone()

    decay_db = 20

    # The power of the impulse response in dB
    power = audio ** 2
    energy = torch.flip(torch.cumsum(torch.flip(power, [0]), 0), [0])  # Integration according to Schroeder

    # remove the possibly all zero tail
    i_nz = torch.max(torch.where(energy > 0)[0])
    n = energy[:i_nz]
    db = 10 * torch.log10(n)
    db = db - db[0]

    # -5 dB headroom
    i_5db = torch.min(torch.where(-5 - db > 0)[0])
    e_5db = db[i_5db]
    t_5db = i_5db / fs

    # after decay
    i_decay = torch.min(torch.where(-5 - decay_db - db > 0)[0])
    t_decay = i_decay / fs

    # compute the decay time
    decay_time = t_decay - t_5db
    est_rt60 = (60 / decay_db) * decay_time

    return est_rt60

def hilbert(x): #hilbert transform
    N = x.shape[1]
    Xf = torch.fft.fft(x, n=None, dim=-1)
    h = torch.zeros(N)
    if N % 2 == 0:
        h[0] = h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N + 1)//2] = 2
    x = torch.fft.ifft(Xf * h)
    return x


def spectral_centroid(x): #calculate the spectral centroid "brightness" of an audio input
    Xf = torch.abs(torch.fft.fft(x,n=None,dim=-1)) #take fft and abs of x
    norm_Xf = Xf / sum(sum(Xf))  # like probability mass function
    norm_freqs = torch.linspace(0, 1, Xf.shape[1])
    spectral_centroid = sum(sum(norm_freqs * norm_Xf))
    return spectral_centroid


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=numpy.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (numpy.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = numpy.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = numpy.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=numpy.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = numpy.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = numpy.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=numpy.uint8)
    else:
        cmap = numpy.zeros((N, 3), dtype=numpy.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (numpy.uint8(str_id[-1]) << (7-j))
                g = g ^ (numpy.uint8(str_id[-2]) << (7-j))
                b = b ^ (numpy.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
