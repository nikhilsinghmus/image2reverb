import torch
import torchvision.transforms
import torchaudio


M_PI = 3.14159265358979323846264338


class STFT(torch.nn.Module):
    def __init__(self, window_size=1024, hop_length=256, window="hann"):
        super().__init__()
        self._w_size = window_size
        self._h_length = hop_length
        d = {"hann": torch.hann_window}
        self._w = d[window](self._w_size) # Window table
        self._n = torchvision.transforms.Normalize(6, 8).cuda()

    def transform(self, audio):
        s = torch.stft(audio, self._w_size, self._h_length, window=self._w, return_complex=True).squeeze()[:-1,:] # Get STFT and trim Nyquist bin
        s = torch.log(s + 1e-6)
        return self._n(torch.abs(s.unsqueeze(0))) # Magnitude

    def inverse(self, spec):
        s = torch.cat((torch.exp(spec), torch.zeros(1, spec.shape[1]).cuda()))
        s = (s * self._n.std) + self._n.mean
        random_phase = torch.Tensor(s.shape).uniform_(-M_PI, M_PI).cuda()
        f = s * (torch.cos(random_phase) + (1.j * torch.sin(random_phase)))
        audio = torch.istft(f, self._w_size, self._h_length, window=self._w.cuda()) # Audio output
        return audio/torch.abs(audio).max()
