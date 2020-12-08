import os
import numpy
import argparse
import joblib
import soundfile
import sklearn
import sklearn.neural_network
import librosa
from model.networks import Encoder
from model.data_loader import CreateDataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="room2reverb", help="Name of the experiment. It decides where to store samples and models.")
    parser.add_argument("--encoder_path", type=str, default="resnet50_places365.pth.tar", help="Path to pre-trained Encoder ResNet50 model.")
    parser.add_argument("--dataset", type=str, default="room2reverb", help="Name of dataset located in the dataset folder.")
    parser.add_argument("--resize_or_crop", type=str, default="scale_width_and_crop", help="Scaling and cropping of images at load time.")
    parser.add_argument("--use_embeddings", action="store_true", help="Use embeddings from pre-trained Places365 model. Default uses images directly.")
    parser.add_argument("--spectrogram", type=str, default="mel", help="Path to pretrained model.")
    parser.add_argument("--output", type=str, default="room2reverb_testbaseline", help="Output directory for examples.")
    parser.add_argument("--verbose", action="store_true", help="Print details.")
    parser.add_argument("--n_samples", type=int, default=float("inf"), help="Max number of samples to process.")
    parser.add_argument("--preload_data", action="store_true", help="Load data from .npy files instead of sources. Directory should contain labels_[train/test].npy and spec_[train/text].npy.")
    args = parser.parse_args()

    args.serial_batches = False
    args.nThreads = 2
    args.max_dataset_size = float("inf")
    args.dataroot = os.path.join("./datasets", args.name)
    args.phase = "train"
    args.isTrain = True
    args.loadSize = 512
    args.fineSize = 224
    args.no_flip = True
    args.batchSize = 1

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # Load the training data
    spec, labels, _, _, model = load_data(args)

    # Set up baseline model
    n_latent = 100

    print("Training model.")
    if args.preload_data:
        m = joblib.load("estimator.joblib")
    else:
        m = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(n_latent), activation="tanh", solver="adam", learning_rate_init=1e-4, max_iter=200, verbose=args.verbose, random_state=2)
        m.fit(labels, spec)
        joblib.dump(m, "estimator.joblib")

    # Load the test data
    args.phase = "test"
    spec, labels, paths, scaler, _ = load_data(args, model)

    output = m.predict(labels)
    output = [s.reshape(128, 256) for s in scaler.inverse_transform(output)]

    numpy.save("output", output)

    for i, example in enumerate(output):
        y = librosa.feature.inverse.mel_to_audio(numpy.exp(example) - 1e-8)
        example_id = os.path.basename(paths[i])
        example_id = example_id[:example_id.rfind("_")]
        soundfile.write(os.path.join(args.output, "%s_output.wav" % example_id), y, 22050)


def load_data(args, model=None):
    data_loader = CreateDataLoader(args)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print("#images = %d" % dataset_size)

    spec = []
    labels = []
    paths = []
    print("Loading data.")
    if args.preload_data:
        spec = numpy.load("spec_%s.npy" % args.phase)
        labels = numpy.load("labels_%s.npy" % args.phase)
        paths = open("paths_%s.txt" % args.phase).read().split("\n")
    else:
        for i, d in enumerate(dataset):
            if i >= args.n_samples:
                break

            if args.verbose:
                print("Loading example no. %d." % (i + 1))

            spec.append(d["image"].squeeze().detach().numpy().flatten())
            labels.append(d["label"].squeeze().detach().numpy().flatten() if not args.use_embeddings else d["label"])
            paths.append(d["path"][0])

        if args.use_embeddings:
            print("Encoding images.")
            model = Encoder(args.encoder_path, device="cpu") if not model else model
            labels = [model(x).squeeze().detach().numpy() for x in labels]

        numpy.save("spec_" + args.phase, spec)
        numpy.save("labels_" + args.phase, labels)
        with open("paths_%s.txt" % args.phase, "w") as outfile:
            outfile.write("\n".join(paths))

    print("Scaling data.")
    scaler = sklearn.preprocessing.StandardScaler().fit(spec) if not args.preload_data else joblib.load("scaler_%s.joblib" % args.phase)
    joblib.dump(scaler, "scaler_%s.joblib" % args.phase)
    spec = scaler.transform(spec)
    return spec, labels, paths, scaler, model


if __name__ == "__main__":
    main()
