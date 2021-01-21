import os
import json
import argparse
import numpy
import torch
import seaborn
import soundfile
import matplotlib
from pytorch_lightning import Trainer, loggers
from image2reverb.model import Image2Reverb
from image2reverb.dataset import Image2ReverbDataset
from matplotlib import pyplot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--encoder_path", type=str, default="resnet50_places365.pth.tar", help="Path to pre-trained Encoder ResNet50 model.")
    parser.add_argument("--depthmodel_path", type=str, default="mono_odom_640x192", help="Path to pre-trained depth (from monodepth2) encoder and decoder models.")
    parser.add_argument("--dataset", type=str, default="./datasets/image2reverb", help="Dataset path.")
    parser.add_argument("--model", type=str, default=None, help="Path to pretrained model.")
    parser.add_argument("--spectrogram", type=str, default="stft", help="Spectrogram type.")
    parser.add_argument("--test_dir", type=str, default=None, help="Dir for test examples.")
    parser.add_argument("--version", type=str, default=None, help="Experiment version.")
    parser.add_argument("--no_depth", action="store_true", help="Don't apply the pre-trained depth model.")
    parser.add_argument("--no_places", action="store_true", help="Don't apply the pre-trained encoder model.")
    parser.add_argument("--constant_depth", type=float, default=None, help="Set depth to constant.")
    parser.add_argument("--n_test", type=float, default=1.0, help="Percentage of test set or the number of test examples.")
    args = parser.parse_args()

    if args.no_places:
        args.encoder_path = None

    if args.no_depth:
        args.depthmodel_path = None

    if not args.test_dir:
        args.test_dir = "image2reverb_%stest/" % args.version

    # Data loading
    cuda = torch.cuda.is_available()
    test_set = Image2ReverbDataset(args.dataset, "test", args.spectrogram)
    test_dataset = torch.utils.data.DataLoader(test_set, num_workers=8, batch_size=args.batch_size) # For now, to test
    
    # Store the test examples
    if not os.path.isdir(args.test_dir):
        os.makedirs(args.test_dir)
    
    def test_fn(examples, t60, spectrograms, audio, input_images, input_depthmaps):
        t60_err = numpy.array(t60) * 100
        numpy.save(os.path.join(args.test_dir, "t60_err"), t60_err)
        
        pyplot.figure(figsize=(4, 5))
        matplotlib.rcParams["font.sans-serif"] = "Avenir"
        matplotlib.rcParams["font.family"] = "sans-serif"
        seaborn.boxplot(y=t60_err)
        pyplot.savefig(os.path.join(args.test_dir, "t60.png"))
        
        t60_d = {}
        for i, example in enumerate(examples):
            print("Processing example %d: %s." % (i, example))
            d = os.path.join(args.test_dir, example)
            if not os.path.isdir(d):
                os.makedirs(d)
            pyplot.imsave(os.path.join(d, "spec.png"), spectrograms[i])
            soundfile.write(os.path.join(d, "%s.wav" % example), audio[i], 22050)
            pyplot.imsave(os.path.join(d, "input.png"), input_images[i])
            pyplot.imsave(os.path.join(d, "depth.png"), input_depthmaps[i])

            t60_d[example] = t60[i]
        
        with open(os.path.join(args.test_dir, "t60.json"), "w") as json_file:
            json.dump(t60_d, json_file, indent=4)
        

    # Main model
    model = Image2Reverb(args.encoder_path, args.depthmodel_path, constant_depth=args.constant_depth, test_callback=test_fn)
    m = torch.load(args.model, map_location=model.device)
    model.load_state_dict(m["state_dict"])
    
    # Model training
    trainer = Trainer(gpus=1, limit_test_batches=args.n_test)
    trainer.test(model, test_dataset)


if __name__ == "__main__":
    main()
