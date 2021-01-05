import os
import argparse
import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from image2reverb.model import Image2Reverb
from image2reverb.dataset import Image2ReverbDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus", type=int, default=1, help="How many GPUs to train with.")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints_image2reverb", help="Model location.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--encoder_path", type=str, default="resnet50_places365.pth.tar", help="Path to pre-trained Encoder ResNet50 model.")
    parser.add_argument("--depthmodel_path", type=str, default="mono_odom_640x192", help="Path to pre-trained depth (from monodepth2) encoder and decoder models.")
    parser.add_argument("--dataset", type=str, default="./datasets/image2reverb", help="Dataset path.")
    parser.add_argument("--niter", type=int, default=100, help="Number of training iters.")
    parser.add_argument("--from_pretrained", type=str, default=None, help="Path to pretrained model.")
    parser.add_argument("--spectrogram", type=str, default="stft", help="Spectrogram type.")
    parser.add_argument("--d_threshold", type=float, default=None, help="Value over which discriminator weights will be updated by optimizer.")
    parser.add_argument("--version", type=str, default=None, help="Experiment version.")
    parser.add_argument("--no_depth", action="store_true", help="Don't apply the pre-trained depth model.")
    parser.add_argument("--no_places", action="store_true", help="Don't load Places365 weights for Encoder.")
    parser.add_argument("--no_t60p", action="store_true", help="Don't apply the T60-style objective term.")
    args = parser.parse_args()

    if args.no_places:
        args.encoder_path = None
        
    if args.no_depth:
        args.depthmodel_path = None

    # Model dir
    folder = args.checkpoints_dir
    if not os.path.isdir(folder):
        os.makedirs(folder)

    cuda = torch.cuda.is_available()
    train_set = Image2ReverbDataset(args.dataset, "train", args.spectrogram)
    val_set = Image2ReverbDataset(args.dataset, "val", args.spectrogram)

    train_dataset = torch.utils.data.DataLoader(train_set, shuffle=True, num_workers=8, pin_memory=cuda, batch_size=args.batch_size)
    val_dataset = torch.utils.data.DataLoader(val_set, num_workers=8, batch_size=args.batch_size) # For now, to test

    # Main model
    model = Image2Reverb(args.encoder_path, args.depthmodel_path, d_threshold=args.d_threshold, t60p=not args.no_t60p)
    
    # Model training
    logger = loggers.TensorBoardLogger(
        args.checkpoints_dir,
        version=args.version
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoints_dir, args.version),
        filename="image2reverb_{epoch:04d}.ckpt",
        period=10,
        save_top_k=-1,
        verbose=True,
    )
    
    trainer = Trainer(
        gpus=args.n_gpus if cuda else None,
        auto_select_gpus=True,
        accelerator="ddp" if cuda else None,
        auto_scale_batch_size="binsearch",
        benchmark=True,
        max_epochs=args.niter,
        resume_from_checkpoint=args.from_pretrained,
        default_root_dir=args.checkpoints_dir,
        callbacks=[checkpoint_callback],
        logger=logger
    )
    
    trainer.fit(model, train_dataset, val_dataset)


if __name__ == "__main__":
    main()
