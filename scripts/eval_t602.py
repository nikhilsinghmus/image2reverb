import sys
import os
import numpy
import scipy.stats
import soundfile
import pyroomacoustics


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    files = (os.path.join(output_dir, f) for f in os.listdir(output_dir) if os.path.splitext(f)[1] == ".wav")
    t60_mae = []
    mfcc_corr = []
    for f in files:
        print(f)
        f_input = os.path.join(input_dir, os.path.basename(f.replace("output", "img")))
        try:
            t60_d, a, b = compare_t60(f, f_input)
            print("%.2f%%: ================> %.2fs %.2fs" % (t60_d * 100, a, b))
            t60_mae.append(t60_d)
        except:
            print("Error.")
    numpy.save("t60", t60_mae)
    print(scipy.stats.describe(t60_mae))


def compare_t60(a, b):
    a, sr = soundfile.read(a)
    b, sr2 = soundfile.read(b)
    t_a = pyroomacoustics.experimental.rt60.measure_rt60(a, sr)
    t_b = pyroomacoustics.experimental.rt60.measure_rt60(b, sr2, rt60_tgt=t_a)
    return ((t_b - t_a)/t_a), t_a, t_b


if __name__ == "__main__":
    main()