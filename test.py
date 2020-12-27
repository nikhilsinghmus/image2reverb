import os
import shutil
import argparse
import torch
import soundfile
from collections import OrderedDict
from model import util
from model.stft import STFT
from model.mel import LogMel
from model.data_loader import CreateDataLoader
from model.model import Room2Reverb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="room2reverb", help="Name of the experiment. It decides where to store samples and models.")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs: e.g. 0  0,1,2, 0,2. use -1 for CPU")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Model location.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--encoder_path", type=str, default="resnet50_places365.pth.tar", help="Path to pre-trained Encoder ResNet50 model.")
    parser.add_argument("--dataset", type=str, default="room2reverb", help="Name of dataset located in the dataset folder.")
    parser.add_argument("--resize_or_crop", type=str, default="scale_width_and_crop", help="Scaling and cropping of images at load time.")
    parser.add_argument("--n_test", type=int, default=100, help="Number of test examples.")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate of output.")
    parser.add_argument("--model", type=str, default="latest", help="Which model/checkpoint to load.")
    parser.add_argument("--spectrogram", type=str, default="stft", help="Spectrogram type.")
    args = parser.parse_args()
    args.batchSize = args.batch_size
    args.serial_batches = False
    args.nThreads = 2
    args.max_dataset_size = float("inf")
    args.dataroot = os.path.join("./datasets", args.name)
    args.phase = "test"
    args.isTrain = True
    args.loadSize = 512
    args.fineSize = 224
    args.no_flip = True

    model = Room2Reverb(args.encoder_path)
    state_dict = torch.load(os.path.join(args.checkpoints_dir, "%s_net_G.pth" % args.model))
    state_dict = {k.replace("module.", ""):v for k, v in state_dict.items()}
    model.load_generator(state_dict)
    
    stft = LogMel() if args.spectrogram == "mel" else STFT()
    folder = args.name + "_test"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    
    data_loader = CreateDataLoader(args)
    dataset = data_loader.load_data()
    
    for i, data in enumerate(dataset):
        if i >= args.n_test:
            break
        generated = model.inference(data["label"].cuda())
        img_path = data["path"][0]
        print("Processing %s." % img_path)
        audio = stft.inverse(generated.squeeze())
        img_outpath = os.path.join(folder, os.path.basename(img_path)).replace("label", "input") 
        audio_path = os.path.splitext(img_outpath)[0].replace("input", "output") + ".wav"
        shutil.copy2(img_path, img_outpath)
        soundfile.write(audio_path, audio, args.sr)

if __name__ == "__main__":
    main()
