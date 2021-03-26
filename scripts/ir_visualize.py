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
    output = "output/images/%s_waveform.png" % f
    src_output = "output/images/%s_input.jpg" % f
    y, sr = soundfile.read(example)

    shutil.copy2(src, src_output)

    y /= numpy.abs(y).max()
    y = y[numpy.where(numpy.abs(y) > 0.01)[0][0]:]
    g = librosa.display.waveplot(y, sr=sr, antialiased=True)
    ax = g.get_figure().axes[0]
    pyplot.xticks(numpy.arange((len(y)//sr) + 1))
    pyplot.yticks(numpy.array([y.min(), 0, y.max()]).round(0))
    ax.tick_params(left=False, bottom=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 5))
    pyplot.plot(len(y)/sr, -0.05, ">k", transform=ax.get_xaxis_transform(), clip_on=False)
    pyplot.savefig(output, bbox_inches="tight")


def set_style():
    pyplot.figure(figsize=(5, 2))
    matplotlib.rcParams["font.sans-serif"] = "Avenir"
    matplotlib.rcParams["font.family"] = "sans-serif"

if __name__ == "__main__":
    main()