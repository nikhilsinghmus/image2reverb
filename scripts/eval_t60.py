import sys
import os
import numpy
import scipy.stats
import scipy.spatial
import soundfile
import librosa
import acoustics


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    bands = acoustics.standards.iec_61672_1_2013.NOMINAL_OCTAVE_CENTER_FREQUENCIES[2:-4]
    files = (os.path.join(output_dir, f) for f in os.listdir(output_dir) if os.path.splitext(f)[1] == ".wav")
    t60_mae = []
    mfcc_corr = []
    for f in files:
        print(f)
        f_input = os.path.join(input_dir, os.path.basename(f.replace("output", "img")))
        t60_d = compare_t60(f, f_input, bands)
        mfcc_s, p_val = compare_mfcc(f, f_input)
        print(os.path.basename(f), os.path.basename(f_input), "%.2fs %.2f %.2f" % (t60_d, mfcc_s, p_val))
        t60_mae.append(t60_d)
        mfcc_corr.append(mfcc_s)
    print(scipy.stats.describe(t60_mae))
    print(scipy.stats.describe(mfcc_corr))


def compare_t60(a, b, bands):
    t60_a = acoustics.room.t60_impulse(a, bands)
    t60_b = acoustics.room.t60_impulse(b, bands)
    return numpy.mean(numpy.abs((t60_a - t60_b)))


def compare_mfcc(a, b):
    a, sr = soundfile.read(a)
    b, sr2 = soundfile.read(b)
    assert sr == sr2
    m1 = librosa.feature.mfcc(a)[0,:255].flatten()
    m2 = librosa.feature.mfcc(b)[0,:255].flatten()
    return scipy.stats.spearmanr(m1, m2)


if __name__ == "__main__":
    main()