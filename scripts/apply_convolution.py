import sys
import numpy
import scipy.signal
import soundfile
import librosa


def main():
    audio, reverb, output = sys.argv[1:]
    y, sr = librosa.load(audio)
    r = apply_reverb(y, reverb)
    soundfile.write(output, r, sr)


def apply_reverb(signal: numpy.ndarray, path: str, mix: float = 0.02) -> numpy.ndarray:
    reverb, _ = soundfile.read(path)
    signal = numpy.concatenate((signal, numpy.zeros(reverb.shape)))
    reverb /= numpy.max(numpy.abs(reverb))
    signal += scipy.signal.oaconvolve(signal, reverb, "full")[:len(signal)] * mix
    return signal


if __name__ == "__main__":
    main()