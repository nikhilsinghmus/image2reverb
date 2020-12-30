import os
import soundfile
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from .stft import STFT
from .mel import LogMel


F_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG",
    ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP", ".tiff", ".wav", ".WAV", ".aif", ".aiff", ".AIF", ".AIFF"
]


def is_image_audio_file(filename):
    return any(filename.endswith(extension) for extension in F_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory." % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_audio_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class Image2ReverbDataset(Dataset):
    def __init__(self, dataroot, phase="train", spec="stft"):
        self.root = dataroot
        self.stft = LogMel() if spec == "mel" else STFT()

        ### input A (images)
        dir_A = "_A"
        self.dir_A = os.path.join(self.root, phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (audio)
        dir_B = "_B"
        self.dir_B = os.path.join(self.root, phase + dir_B)  
        self.B_paths = sorted(make_dataset(self.dir_B))
      
    def __getitem__(self, index):
        if index > len(self):
            return None
        ### input A (images)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        A_tensor = t(A.convert("RGB"))

        ### input B (audio)
        B_path = self.B_paths[index]
        B, _ = soundfile.read(B_path)
        B_spec = self.stft.transform(B)

        return B_spec, A_tensor, (B_path, A_path)

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return "Image2Reverb"
