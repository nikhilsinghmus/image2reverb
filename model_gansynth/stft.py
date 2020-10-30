import torch
import torchvision.transforms
import torchaudio


M_PI = 3.14159265358979323846264338


class STFT(torch.nn.Module):
    def __init__(self, window_size=1024, hop_length=256, window="hann"):
        super().__init__()
        self._w_size = window_size
        self._h_length = hop_length
        self._w = {"hann": torch.hann_window}[window](self._w_size) # Window table
        self._n = torchvision.transforms.Normalize((0.5, 0.5), (0.5, 0.5))

    def transform(self, audio):
        s = torch.stft(audio, self._w_size, self._h_length, window=self._w, return_complex=True).squeeze()[:-1,:] # Get STFT and trim Nyquist bin
        m = torch.abs(s) # Magnitude
        phase_angle = torch.angle(s) # Phase angle
        i_f = phase_angle[:,1:] - phase_angle[:,:-1] # Finite difference
        i_f = torch.cat((phase_angle[:,:1], i_f), axis=1)
        i_f = torch.where(i_f > M_PI, i_f - 2 * M_PI, i_f)
        i_f = torch.where(i_f < -M_PI, i_f + 2 * M_PI, i_f)
        m = torch.log(m + 1.0e-6)
        return self._n(torch.stack((m, i_f))) # (2, 512, 512) output (log magnitude)

    def inverse(self, spec):
        m, i_f = spec # (2, 512, 512) input
        m = torch.exp(m) # Linear magnitude
        phase_angle = torch.cumsum(i_f, 1) * M_PI
        phase = torch.cos(phase_angle) + (torch.sin(phase_angle) * 1j) # Cartopol, basically
        s = torch.cat((m * phase, torch.zeros((1, m.shape[1]))), axis=0) # Zero-pad for Nyquist bin
        audio = torch.istft(s, self._w_size, self._h_length, window=self._w) # Audio output
        return audio/torch.abs(audio).max()
