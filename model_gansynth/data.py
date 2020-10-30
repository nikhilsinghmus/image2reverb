import os
import torchvision
import torchaudio
from PIL import Image
from .stft import STFT


def make_dataset(dir): # Iteratively make the dataset (adapted from Pix2PixHD code)
    f = []

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if os.path.splitext(fname)[1].lower() in (".jpg", ".jpeg", ".png", ".wav"):
                path = os.path.join(root, fname)
                f.append(path)

    return f


class Dataset:
    def __init__(self, dataset, phase="train"): 

        dir_A = "_A" # Images
        self.dir_A = os.path.join(dataset, phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        dir_B = "_B" # Audio
        self.dir_B = os.path.join(dataset, phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths)
        self.stft = STFT()
      
    def __getitem__(self, index):        
        A_path = self.A_paths[index]
        A = Image.open(A_path)
        A_tensor = torchvision.transforms.ToTensor()(A).unsqueeze(0) # Transform image to tensor and make a batch

        B_path = self.B_paths[index]
        B, _ = torchaudio.load(B_path)
        B_spec = self.stft.transform(B).unsqueeze(0).cuda() # Compute the spectral representation

        return A_tensor, B_spec

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return "Dataset"
