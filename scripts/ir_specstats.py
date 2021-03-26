import sys
import os
import numpy
import soundfile
import librosa


def main():
    data_dir = "../datasets/room2reverb/test_B/"
    u = []
    std = []
    mins = []
    maxes = []
    files = os.listdir(data_dir)
    n = len(files)
    for i, f in enumerate(files):
        if not f.endswith(".wav"):
            continue
        print("Processing %d of %d." % (i + 1, n))
        f_path = os.path.join(data_dir, f)
        y, sr = soundfile.read(f_path)
        y /= numpy.abs(y).max()
        m = numpy.abs(librosa.stft(y, 1024, 256))[:-1,:]
        m = numpy.log(m + 1e-8)
        u.append(m.mean())
        std.append(m.std())
        mins.append(m.min())
        maxes.append(m.max())

        print("Mean: %.2f." % m.mean())
        print("Std: %.2f." % m.std())
        print("Min: %.2f." % m.min())
        print("Max: %.2f." % m.max())

    print("\n\nDataset stats:\n")
    print("Mean: %.4f." % numpy.mean(u))
    print("Std: %.4f." % numpy.mean(std))
    print("Min: %.4f." % numpy.mean(mins))
    print("Max: %.4f." % numpy.mean(maxes))

if __name__ == "__main__":
    main()