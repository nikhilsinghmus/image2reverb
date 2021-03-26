import sys
import os
import numpy
import scipy.spatial
import soundfile
import sklearn.preprocessing
import umap
import acoustics
from matplotlib import pyplot
from lapjv import lapjv
from PIL import Image


def main():
    audio_dir = sys.argv[1]
    image_dir = sys.argv[2]
    files = [os.path.join(audio_dir, f) for f in sorted(os.listdir(audio_dir)) if os.path.splitext(f)[1] == ".wav"]
    images = (os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if os.path.splitext(f)[1]  in [".png", ".jpg"])

    load_audio = True
    audio = []
    if os.path.isfile("audio.npy"):
        audio = numpy.load("audio.npy")
        load_audio = False
    imgs = []
    bands = acoustics.standards.iec_61672_1_2013.NOMINAL_OCTAVE_CENTER_FREQUENCIES[2:-4]
    print("Data loading.")
    m = len(files)
    for i, (audio_f, image_f) in enumerate(zip(files, images)):
        print("Processing %d of %d." % (i + 1, m))
        assert os.path.basename(audio_f[:audio_f.rfind("_")]) == os.path.basename(image_f[:image_f.rfind("_")])
        imgs.append(Image.open(image_f).convert("RGB"))
        if not load_audio:
            continue
        audio.append(acoustics.room.t60_impulse(audio_f, bands))

    if load_audio:
        audio = numpy.array(audio)
        numpy.save("audio", audio)

    w = 26
    h = 42

    print("UMAP to 2 dimensions.")
    f = generate_umap(audio)

    print("Make grid and store.")
    save_grid(imgs, f, w, h)


def generate_umap(X):
    u = umap.UMAP(n_components=2)
    X_2d = u.fit_transform(X)
    X_2d -= X_2d.min(axis=0)
    X_2d /= X_2d.max(axis=0)
    return X_2d


def save_grid(img_collection, X_2d, w, h, out_res=224):
    grid = numpy.dstack(numpy.meshgrid(numpy.linspace(0, 1, w), numpy.linspace(0, 1, h))).reshape(-1, 2)
    cost_matrix = scipy.spatial.distance.cdist(grid, X_2d, "sqeuclidean").astype(numpy.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    r, c, _ = lapjv(cost_matrix)
    grid_jv = grid[c]
    out = numpy.zeros((w*out_res, h*out_res, 3))

    for pos, img in zip(grid_jv, img_collection):
        w_range = int(numpy.floor(pos[0] * (w - 1) * out_res))
        h_range = int(numpy.floor(pos[1] * (h - 1) * out_res))
        out[w_range:w_range + out_res,h_range:h_range + out_res,:]  = numpy.array(img).astype(numpy.float32)/255

    pyplot.imsave("out.jpg", out)


if __name__ == "__main__":
    main()