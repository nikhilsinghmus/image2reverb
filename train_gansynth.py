import os
import argparse
import time
import torch
import torchaudio
from model_gansynth.model import Room2Reverb
from model_gansynth.data import Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="room2reverb", help="Name of the experiment. It decides where to store samples and models.")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs: e.g. 0  0,1,2, 0,2. use -1 for CPU")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Model location.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--encoder_path", type=str, default="resnet50_places365.pth.tar", help="Path to pre-trained Encoder ResNet50 model.")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--dataset", type=str, default="room2reverb", help="Name of dataset located in the dataset folder.")
    args = parser.parse_args()

    d_path = os.path.join("./datasets", args.dataset)
    dataset = Dataset(d_path)

    model = Room2Reverb(args.encoder_path)
    epoch_iter = 0

    for epoch in range(args.n_epochs):
        epoch_start_time = time.time()
        if epoch != 0:
            epoch_iter = epoch_iter % dataset.dataset_size

        for i, (img, spec) in enumerate(dataset, start=epoch_iter):
            model.train_step(epoch, spec, img, args.name + "_models", i - epoch_iter)
            print("G ", model.G_loss.item())
            print("D ", model.D_loss.item())
            print("------------------------")


if __name__ == "__main__":
    main()
