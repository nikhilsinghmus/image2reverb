import numpy
import torch
import torchaudio
import torch.fft

def estimate_t60(audio, sr):
    init = -5.0
    end = -35.0

    bands = torch.FloatTensor([125, 250, 500, 1000, 2000, 4000])
    t60 = torch.zeros(bands.shape[0])

    for band in range(bands.shape[0]):
        # Filtering signal
        filtered_signal = torchaudio.functional.bandpass_biquad(audio, sr, bands[band])
        abs_signal = torch.abs(filtered_signal) / torch.max(torch.abs(filtered_signal))

        # Schroeder integration
        sch = torch.flip(torch.cumsum(torch.flip(abs_signal, [0]) ** 2, 0), [0])
        sch_db = 10.0 * torch.log10(sch / torch.max(sch))

        sch_init = sch_db[torch.abs(sch_db - init).argmin()]
        sch_end = sch_db[torch.abs(sch_db - end).argmin()]
        init_sample = torch.where(sch_db == sch_init)[0][0]
        end_sample = torch.where(sch_db == sch_end)[0][0]
        t60[band] = 2 * (end_sample - init_sample)
    return t60

def hilbert(x): #hilbert transform
    N=x.shape[1]
    Xf = torch.fft.fft(x,n=None,dim=-1)
    h=torch.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    x = torch.fft.ifft(Xf * h)
return x
