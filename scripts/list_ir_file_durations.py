import os
import glob
import soundfile
import settings


def main():
    l = []
    for f in glob.glob(os.path.join(settings.dataset_dir, "*/*.wav")):
        y, fs = soundfile.read(f)
        l.append((os.path.basename(f), len(y)/fs))
    l.sort(key=lambda f : -f[1])
    with open("ir_lengths.txt", "w") as outfile:
        outfile.write("\n".join(("%s, %.4f" % (f, d) for f, d in l)))


if __name__ == "__main__":
    main()