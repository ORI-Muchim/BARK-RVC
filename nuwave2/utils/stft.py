import torch
import torch.nn as nn
import torch.nn.functional as F

class STFTMag(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)

    @torch.no_grad()
    def forward(self, x):
        T = x.shape[-1]
        window = self.window.to(x.device)
        stft = torch.stft(x, self.nfft, self.hop, window=window, return_complex=True)
        mag = torch.norm(stft, p=2, dim=-1)  # [B, F, TT]
        return mag

