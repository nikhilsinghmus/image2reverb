import sys
import os
import shutil
import numpy
import cv2
import skimage.metrics


def main():
    get_files = lambda d : [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg")]
    f1, f2 = map(get_files, sys.argv[1:3])

    if not os.path.isdir("compare"):
        os.makedirs("compare")

    make_fname = lambda s, i : os.path.join("compare", os.path.basename(s).replace(".jpg", "_%d_out.jpg" % i))
    m = max(len(f1), len(f2))
    for i, imgs in enumerate(zip(f1, f2)):
        print("Processing %d of %d." % (i + 1, m))
        img1, img2 = map(lambda f : cv2.cvtColor(cv2.imread(f, 1), cv2.COLOR_BGR2GRAY), imgs)
        if skimage.metrics.structural_similarity(img1, img2, full=True)[0] < 0.98:
            for i, img in enumerate(imgs):
                shutil.copy2(img, make_fname(img, i + 1))



if __name__ == "__main__":
    main()