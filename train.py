import os
import argparse
import torch
from pytorch_lightning import Trainer, loggers
from image2reverb.model import Image2Reverb
from image2reverb.dataset import Image2ReverbDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus", type=int, default=1, help="How many GPUs to train with.")
    parser.add_argument("--checkpoints_dir", type=str, default="./image2reverb_checkpoints", help="Model location.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--encoder_path", type=str, default="resnet50_places365.pth.tar", help="Path to pre-trained Encoder ResNet50 model.")
    parser.add_argument("--depthmodel_path", type=str, default="mono_odom_640x192", help="Path to pre-trained depth (from monodepth2) encoder and decoder models.")
    parser.add_argument("--dataset", type=str, default="./datasets/room2reverb", help="Dataset path.")
    parser.add_argument("--niter", type=int, default=200, help="Number of training iters.")
    parser.add_argument("--from_pretrained", type=str, default=None, help="Path to pretrained model.")
    parser.add_argument("--spectrogram", type=str, default="stft", help="Spectrogram type.")
    args = parser.parse_args()

    # Model dir
    folder = args.checkpoints_dir
    if not os.path.isdir(folder):
        os.makedirs(folder)

    cuda = torch.cuda.is_available()
    train_set = Image2ReverbDataset(args.dataset, "train", args.spectrogram)
    val_set = Image2ReverbDataset(args.dataset, "test", args.spectrogram)

    train_dataset = torch.utils.data.DataLoader(train_set, shuffle=True, num_workers=8, pin_memory=cuda, batch_size=args.batch_size)
    val_dataset = torch.utils.data.DataLoader(val_set, num_workers=8, batch_size=args.batch_size) # For now, to test

    # Main model
    model = Image2Reverb(args.encoder_path, args.depthmodel_path)
    trainer = Trainer(gpus=args.n_gpus if cuda else None, accelerator="ddp" if cuda else None, auto_scale_batch_size="binsearch", benchmark=True, limit_val_batches=0.25, max_epochs=args.niter, resume_from_checkpoint=args.from_pretrained, weights_save_path=args.checkpoints_dir, default_root_dir=args.checkpoints_dir, num_sanity_val_steps=0)
    trainer.fit(model, train_dataset, val_dataset)


if __name__ == "__main__":
    main()
