import sys
import os
import shutil
import numpy
import soundfile
import librosa
import librosa.display
import matplotlib
from matplotlib import pyplot


def main():
    src_dir = "../datasets/room2reverb/test_A/"
    data_dir = "../datasets/room2reverb/test_B/"

    if not os.path.isdir("output/images/"):
        os.makedirs("output/images/")

    set_style()

    f = sys.argv[1]
    example = os.path.join(data_dir, "%s_img.wav" % f)
    src = os.path.join(src_dir, "%s_label.jpg" % f)
    output = "output/images/%s_spec.png" % f
    src_output = "output/images/%s_input.jpg" % f
    y, sr = soundfile.read(example)

    shutil.copy2(src, src_output)

    y /= numpy.abs(y).max()
    t = numpy.where(numpy.abs(y) > 0.00001)
    y = y[t[0][0]:t[0][-1]]

    m = librosa.feature.melspectrogram(y)
    m = numpy.log(m + 1e-8)[1:,:]

    fig = pyplot.figure(figsize=(8, 4))
    ax = fig.gca(projection="3d")
    f = librosa.mel_frequencies()[1:, None]
    t = (numpy.linspace(0, 1, m.shape[1]) * (len(y)/sr))[None, :]
    ax.plot_surface(t, f, m, cmap="coolwarm")
    ax.set_ylim(f[-1], f[0])
    ax.set_zticks([])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    bg = (0, 0, 0, 0)
    ax.w_zaxis.line.set_color(bg)
    ax.w_yaxis.line.set_color(bg)
    ax.w_zaxis.set_pane_color(bg)
    ax.w_yaxis.set_pane_color(bg)
    ax.w_xaxis.set_pane_color(bg)
    ax.grid(False)
    # pyplot.show()
    pyplot.savefig(output, bbox_inches="tight")


def set_style():
    pyplot.figure(figsize=(5, 2))
    matplotlib.rcParams["font.sans-serif"] = "Avenir"
    matplotlib.rcParams["font.family"] = "sans-serif"

if __name__ == "__main__":
    main()