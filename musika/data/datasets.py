from typing import Dict
from typing import Optional

import math

import torch
import torchaudio


class RawAudioDataset(torch.utils.data.Dataset):
    """A dataset that yields chunks of a larger audio file.

    Args:
        path (str): The path to the audio file.
        freq (int): Audio sampling frequency of yielded samples.
        window_length (int): Length of a chunk in milliseconds.
        drop_last (bool): Wether to drop the last chunk of the 
                          audio file that may be smaller than `window_length`
        n_fft (int): Size of the spectrograms FFT.
        win_length (int): The window size of the spectrogram.
        hop_length (int): The length of hop between stft windows in the spectrogram.
        pad (int): The value set for padding in the spectrogram.

    Note:
        This dataset yields spectrograms of chunks sourced from a given audio file.
        These chunks are equally sized by the `window_length`.
    """
    def __init__(
        self,
        path: str,
        freq: int = 16_000,
        window_length: float = 760,
        drop_last: bool = True,
        n_fft: int = 256,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = 64,
        pad: int = 0,
    ) -> None:
        self.path = path
        self.freq = freq
        self.window_length = window_length
        self.drop_last = drop_last
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.pad = pad

        waveform, orig_freq = torchaudio.load(
            path, normalize=True)
        resample = torchaudio.transforms.Resample(
            orig_freq=orig_freq,
            new_freq=self.freq)
        self.waveform = resample(waveform)
        del waveform
        del resample

        self.window = int(self.freq * (self.window_length / 1000))
        num_windows = int(self.waveform.shape[-1] / self.window)
        self.num_windows = num_windows - 1 if self.drop_last else num_windows

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            power=None)
        
    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_frame = self.window * idx
        end_frame = self.window * (idx + 1)

        waveform = self.waveform[:, start_frame:end_frame]
        spectrogram = self.spectrogram(waveform)

        return spectrogram