import torch
import torchaudio


class STFT(torch.nn.Module):
    """Spectral representation of audio."""
    def __init__(self, window_size=1024, hop_length=256, window="hann"):
        super().__init__()

        self._w_size = window_size
        self._h_length = hop_length
        self._w = {"hann": torch.hann_window}[window](self._w_size) # Hanning window

    def transform(self, input_data): # STFT (2 channel output)
        return torchaudio.functional.spectrogram(input_data, 0, self._w, self._w_size, self._h_length, self._w_size, None, False).squeeze()[:-1,:]

    def inverse(self, spec): # STFT inverse
        return torchaudio.functional.istft(spec, self._w_size, self._h_length, self._w_size, self._w, False, None, False, True)
