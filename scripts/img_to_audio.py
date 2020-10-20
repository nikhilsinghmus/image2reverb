import sys
import os
import numpy
import soundfile
import librosa
from PIL import Image


def main():
    d = sys.argv[1]
    files = (os.path.join(d, f) for f in os.listdir(d) if os.path.splitext(f)[1] in (".png", ".jpg"))
    for f in files:
        print(f)
        y = img_to_audio(f)
        out_f = os.path.splitext(f)[0] + ".wav"
        soundfile.write(out_f, y, 22050)


def img_to_audio(x):
    img = Image.open(x).convert("L")
    spec = numpy.array(img).astype(numpy.float32)/255
    y = librosa.istft(spec)
    y /= numpy.max(numpy.abs(y))
    return y


if __name__ == "__main__":
    main()