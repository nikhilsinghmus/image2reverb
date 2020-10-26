import os.path
import torchaudio
from PIL import Image
from .base_dataset import BaseDataset, get_params, get_transform, normalize
from .image_folder import make_dataset
from .stft import STFT

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.stft = STFT()

        ### input A (images)
        dir_A = "_A"
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (audio)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = "_B"
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (images)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A.convert("RGB"))

        ### input B (audio)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]
            B, _ = torchaudio.load(B_path)
            B_spec = self.stft.transform(B).permute(2, 0, 1)
        
        inst_tensor = feat_tensor = 0
        input_dict = {"label": A_tensor, "inst": inst_tensor, "image": B_spec, 
                      "feat": feat_tensor, "path": A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return "AlignedDataset"
