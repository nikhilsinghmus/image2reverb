import sys
import os
import numpy
import scipy.spatial
import soundfile
import sklearn.preprocessing
import sklearn.manifold
import acoustics
from matplotlib import pyplot
from lapjv import lapjv
from PIL import Image


def main():
    audio_dir = sys.argv[1]
    image_dir = sys.argv[2]
    files = (os.path.join(audio_dir, f) for f in sorted(os.listdir(audio_dir)) if os.path.splitext(f)[1] == ".wav")
    images = (os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if os.path.splitext(f)[1]  in [".png", ".jpg"])

    audio = []
    imgs = []
    bands = acoustics.standards.iec_61672_1_2013.NOMINAL_OCTAVE_CENTER_FREQUENCIES[2:-4]
    print("Data loading.")
    for audio_f, image_f in zip(files, images):
        assert os.path.basename(audio_f[:audio_f.rfind("_")]) == os.path.basename(image_f[:image_f.rfind("_")])
        audio.append(acoustics.room.t60_impulse(audio_f, bands, rt="edt"))
        imgs.append(Image.open(image_f).convert("RGB"))

    # print("Scaling and PCA to 10 dimensions.")
    audio = numpy.array(audio)
    # audio_input = sklearn.preprocessing.scale(audio)
    # pca_f = sklearn.decomposition.PCA(n_components=10).fit_transform(audio_input)

    w = 26
    h = 42

    print("T-SNE to 2 dimensions.")
    tsne_f = generate_tsne(audio)

    print("Make grid and store.")
    save_tsne_grid(imgs, tsne_f, w, h)


def generate_tsne(X):
    tsne = sklearn.manifold.TSNE(n_components=2)
    X_2d = tsne.fit_transform(X)
    X_2d -= X_2d.min(axis=0)
    X_2d /= X_2d.max(axis=0)
    return X_2d


def save_tsne_grid(img_collection, X_2d, w, h, out_res=224):
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