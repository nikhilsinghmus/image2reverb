import sys
sys.path.append("../")
import os
import argparse
import numpy
import torch
import matplotlib
from matplotlib import pyplot
from model.data_loader import CreateDataLoader
from model.networks import Encoder


def main():
    args = argparse.ArgumentParser().parse_args()
    args.resize_or_crop = "scale_width_and_crop"
    args.spectrogram = "stft"
    args.batchSize = 1
    args.serial_batches = False
    args.nThreads = 2
    args.max_dataset_size = float("inf")
    args.dataroot = "../datasets/room2reverb/"
    args.phase = "test"
    args.isTrain = True
    args.loadSize = 224
    args.fineSize = 224
    args.no_flip = True

    if not os.path.isdir("tmp"):
        os.makedirs("tmp")

    n = Encoder("../resnet50_places365.pth.tar", "../mono_odom_640x192", "cpu")

    data_loader = CreateDataLoader(args)
    dataset = data_loader.load_data()

    matplotlib.rcParams["font.sans-serif"] = "Avenir"
    matplotlib.rcParams["font.family"] = "sans-serif"
    
    for i, data in enumerate(dataset):
        img = data["label"]
        _, x = n(img)
        make_p(x)
        pyplot.savefig("tmp/channels_%s.png" % os.path.basename(data["path"][0][:data["path"][0].rfind("_")]), bbox_inches="tight")


def make_p(x):
    f = x.squeeze().detach().numpy()
    f[:-1,:,:] = (f[:-1,:,:] + 1) * 0.5
    channels = ["R", "G", "B", "Depth"]
    fig, subplots = pyplot.subplots(ncols=4, nrows=1, sharey=True)
    for i in range(3):
        tmp = numpy.zeros((224, 224, 3))
        tmp[:,:,i] = f[i]
        subplots[i].imshow(tmp)
        subplots[i].set_axis_off()
        subplots[i].set_title(channels[i])

    d = f[-1]
    subplots[-1].imshow(d)
    subplots[-1].set_axis_off()
    subplots[-1].set_title(channels[-1])


if __name__ == "__main__":
    main()