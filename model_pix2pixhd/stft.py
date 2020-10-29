import torch
import torchaudio


M_PI = 3.14159265358979323846264338


class STFT(torch.nn.Module):
    def __init__(self, window_size=1024, hop_length=256, window="hann"):
        super().__init__()

        self._w_size = window_size
        self._h_length = hop_length
        self._w = {"hann": torch.hann_window}[window](self._w_size).cuda()

    def transform(self, audio):
        s = torch.stft(audio, self._w_size, self._h_length, return_complex=True).squeeze()[:-1,:]
        m = torch.abs(s)
        phase_angle = torch.angle(s)
        i_f = phase_angle[:,1:] - phase_angle[:,:-1]
        i_f = torch.where(i_f > M_PI, i_f - 2 * M_PI, i_f)
        i_f = torch.where(i_f < M_PI, i_f + 2 * M_PI, i_f)
        i_f = torch.cat((phase_angle[:,:1], i_f), axis=1)
        return torch.stack((torch.log(m), i_f)).cuda()

    def inverse(self, spec):
        m, i_f = spec
        m = torch.exp(m)
        phase_angle = torch.cumsum(i_f, 1)
        phase = torch.cos(phase_angle) + (torch.sin(phase_angle) * 1j)
        s = torch.cat((m * phase, torch.zeros((1, m.shape[1])).cuda()), axis=0)
        return torch.istft(s, self._w_size, self._h_length)
