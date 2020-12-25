import numpy
import torch
import librosa


class LogMel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._eps = 1e-8

    def transform(self, audio):
        m = numpy.abs(librosa.stft(audio/numpy.abs(audio).max()))[:-1,:]
        m = numpy.log(m + self._eps)
        m = (((m - m.min())/(m.max() - m.min()) * 2) - 1)
        return torch.Tensor(m * 0.8).unsqueeze(0)

    def inverse(self, spec):
        s = spec.cpu().detach().numpy()
        s = numpy.exp((((s + 1) * 0.5) * 19.5) - 17.5) - self._eps # Empirical (average) min and max over test set
        y = librosa.istft(s) # Reconstruct audio
        return y/numpy.abs(y).max()
