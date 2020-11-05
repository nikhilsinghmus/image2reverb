import os
import numpy
import torch
import torchaudio
import torch.fft
from PIL import Image

def estimate_t60(audio, sr):
    init = -5.0
    end = -35.0

    audio -= audio.min(1,keepdim=True)[0] # normalize audio to -1:1 because bandpass_biquad clips
    audio /= audio.max(1,keepdim=True)[0]/2
    audio -= 1

    bands = torch.FloatTensor([125, 250, 500, 1000, 2000, 4000])
    t60 = torch.zeros(bands.shape[0])

    for band in range(bands.shape[0]):
        # Filtering signal
        filtered_signal = torchaudio.functional.bandpass_biquad(audio, sr, bands[band])
        analytic_signal = torch.abs(hilbert(filtered_signal)) #absolute value of hilbert transform

        # Schroeder integration
        sch = torch.flip(torch.cumsum(torch.flip(analytic_signal, [0]) ** 2, 0), [0])
        sch_db = 10.0 * torch.log10(sch / torch.max(sch))

        sch_init = sch_db[0,torch.abs(sch_db - init).argmin()]
        sch_end = sch_db[0,torch.abs(sch_db - end).argmin()]
        init_sample = torch.where(sch_db == sch_init)[1][0]
        end_sample = torch.where(sch_db == sch_end)[1][0]
        t60[band] = 2 * (end_sample - init_sample)
    return t60

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
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
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
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
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
