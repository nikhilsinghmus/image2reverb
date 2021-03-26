import numpy
import torch
import librosa


class LogMel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._eps = 1e-8

    def transform(self, audio):
        m = librosa.feature.melspectrogram(audio/numpy.abs(audio).max())
        m = numpy.log(m + self._eps)
        return torch.Tensor(((m - m.mean()) / m.std()) * 0.8).unsqueeze(0)

    def inverse(self, spec):
        s = spec.cpu().detach().numpy()
        s = numpy.exp((s * 5) - 15.96) - self._eps # Empirical mean and standard deviation over test set
        y = librosa.feature.inverse.mel_to_audio(s) # Reconstruct audio
        return y/numpy.abs(y).max()
