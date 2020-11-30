import sys
import os
import subprocess
import numpy
import scipy.signal
import soundfile
import librosa


def main():
    audio = sys.argv[1]
    example = sys.argv[2]
    data_dir = sys.argv[3]
    reverb = os.path.join(data_dir, "%s_output.wav" % example)
    s = audio.split("/")
    output = "output/test_%s_%s" % (example, s[s.index("Anechoic") + 1])
    y, sr = librosa.load(audio)
    r = apply_reverb(y, reverb)
    r /= numpy.max(numpy.abs(r))
    soundfile.write(output + ".wav", r, sr)

    subprocess.run(["ffmpeg", "-loop", "1", "-i", os.path.join(data_dir, "%s_input.jpg" % example), "-i", output + ".wav", "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest", output + ".mp4"])


def apply_reverb(signal: numpy.ndarray, path: str) -> numpy.ndarray:
    reverb, _ = soundfile.read(path)
    signal = numpy.concatenate((signal, numpy.zeros(reverb.shape)))
    reverb /= numpy.max(numpy.abs(reverb))
    reverb[numpy.where(numpy.abs(reverb) < 0.1)] = 0
    mix = 0.01
    signal += scipy.signal.oaconvolve(signal, reverb, "full")[:len(signal)] * mix
    return signal


if __name__ == "__main__":
    main()