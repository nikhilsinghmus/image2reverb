import sys
import numpy
import soundfile
import librosa

def main():
    fs_new = 22050
    length_new = 5.94 # to make it resulting spectrograms 512x512
    for f in sys.argv[1:]:
        y, fs = soundfile.read(f)
        y_22k = librosa.resample(y, fs, fs_new) #resample to 22.05k
        y_22k_pad = librosa.util.fix_length(y_22k, round(fs_new*length_new)) #pad to length_new seconds
        soundfile.write(f, y_22k_pad, fs_new, subtype='PCM_16')
        print("Processed and wrote %s." % f)


if __name__ == "__main__":
    main()
