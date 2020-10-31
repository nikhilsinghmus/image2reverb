import sys
import subprocess
import numpy
import scipy.signal
import soundfile
import librosa


def main():
    audio = sys.argv[1]
    example = sys.argv[2]
    data_dir = sys.argv[3]
    reverb = os.path.join(data_dir, "%s_label_synthesized_image.wav" % example)
    s = audio.split("/")
    output = "output/test_%s_%s" % (example, s[s.index("Anechoic") + 1])
    y, sr = librosa.load(audio)
    r = apply_reverb(y, reverb, r = len(sys.argv) > 3)
    soundfile.write(output + ".wav", r, sr)

    subprocess.run(["ffmpeg", "-loop", "1", "-i", os.path.join(data_dir, "inputs/%s_label_input_label.jpg" % example), "-i", output + ".wav", "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest", output + ".mp4"])


def apply_reverb(signal: numpy.ndarray, path: str, mix: float = 0.02, r = False) -> numpy.ndarray:
    reverb, _ = soundfile.read(path)
    if r:
        reverb = reverb[::-1]
    signal = numpy.concatenate((signal, numpy.zeros(reverb.shape)))
    reverb /= numpy.max(numpy.abs(reverb))
    signal += scipy.signal.oaconvolve(signal, reverb, "full")[:len(signal)] * mix
    return signal


if __name__ == "__main__":
    main()