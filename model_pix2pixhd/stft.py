import torch
import torchaudio

class STFT(torch.nn.Module):
    def __init__(self, window_size=1024, hop_length=256, window="hann"):
        super().__init__()

        self._w_size = window_size
        self._h_length = hop_length
        self._w = {"hann": torch.hann_window}[window](self._w_size).cuda()

    def transform(self, input_data):
        return torchaudio.functional.spectrogram(input_data, 0, self._w, self._w_size, self._h_length, self._w_size, None, False).squeeze()[:-1,:]

    def inverse(self, spec):
        spec = spec.permute(1, 2, 0)
        spec = torch.cat((spec, torch.zeros((1, spec.shape[1], spec.shape[2])).cuda()))
        return torchaudio.functional.istft(spec, self._w_size, self._h_length, self._w_size, self._w, True, None, False, True)
