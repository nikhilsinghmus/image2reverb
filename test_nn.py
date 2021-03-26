import os
import shutil
import json
import argparse
import numpy
import tqdm
import sklearn
import torch
import seaborn
import soundfile
import matplotlib
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
    train_set = Image2ReverbDataset(args.dataset, "train", args.spectrogram)
    test_set = Image2ReverbDataset(args.dataset, "test", args.spectrogram)
    
    # Store the test examples
    if not os.path.isdir(args.test_dir):
        os.makedirs(args.test_dir)

    # Main model
    model = Image2Reverb(args.encoder_path, args.depthmodel_path)
    encoder = model.enc

    embeddings = []
    audio = []
    
    f = "embeddings.npy"
    h = False
    if os.path.isfile(f):
        embeddings = numpy.load(f)
        h = True
    
    f_a = "audio.txt"
    h_a = False
    if os.path.isfile(f_a):
        with open(f_a, "r") as infile:
            audio = infile.read().split("\n")
            h_a = True
    
    if not (h and h_a):
        for _, img, (audio_path, _) in tqdm.tqdm(train_set):
            if not h:
                embeddings.append(encoder(img.unsqueeze(0))[0].detach().cpu().numpy())
            if not h_a:
                audio.append(audio_path)
    
        embeddings = numpy.array(embeddings)
        numpy.save("embeddings", embeddings)

        with open("audio.txt", "w") as outfile:
            outfile.write("\n".join(audio))
    
    embeddings = embeddings[:, 0, :, 0, 0]

    t = sklearn.preprocessing.StandardScaler().fit(embeddings)
    c = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1).fit(t.transform(embeddings), numpy.arange(len(audio)))
    
    test_embeddings = []
    paths = []

    f = "test_embeddings.npy"
    h = False
    if os.path.isfile(f):
        test_embeddings = numpy.load(f)
        h = True
    
    f_a = "paths.npy"
    h_p = False
    if os.path.isfile(f_a):
        paths = numpy.load(f_a).tolist()
        h_p = True
    
    if not (h and h_p):
        for _, img, p_audioimg in tqdm.tqdm(test_set):
            if not h:
                test_embeddings.append(encoder(img.unsqueeze(0))[0].detach().cpu().numpy())
            if not h_p:
                paths.append(p_audioimg)
    
        numpy.save(f_a, paths)
    
        test_embeddings = numpy.array(test_embeddings)
        numpy.save("test_embeddings", test_embeddings)
    
    test_embeddings = test_embeddings[:, 0, :, 0, 0]
        
    l = c.predict(t.transform(test_embeddings))

    for i, n in tqdm.tqdm(enumerate(l)):
        _, path_img = paths[i]
        path_a = audio[n]
        example = os.path.basename(path_img[:path_img.rfind("_")])
        print("Processing example %d: %s." % (i, example))
        d = os.path.join(args.test_dir, example)
        if not os.path.isdir(d):
            os.makedirs(d)
        shutil.copy2(path_a, os.path.join(d, "%s.wav" % example))
        shutil.copy2(path_img, os.path.join(d, "input.png"))


if __name__ == "__main__":
    main()
