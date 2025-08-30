import matplotlib.pyplot as plt
import numpy as np
import roseus.mpl as rs
import torch
import torchaudio.transforms as T
from cycler import cycler
from matplotlib.axes import Axes

from hs_hackathon_drone_acoustics.base import AudioWaveform

# Global plot config - use roseus colormap and extract 5 colors from it for prop cycle.
plt.style.use("dark_background")
plt.rcParams["image.cmap"] = "rs.roseus"
plt.rcParams["axes.prop_cycle"] = cycler(color=[rs.roseus(i) for i in np.linspace(0.4, 1, 5)])


def plot_waveform(waveform: AudioWaveform, axis: Axes) -> None:
    num_samples: int = waveform.data.shape[0]
    time_axis = torch.linspace(0, num_samples / waveform.sample_rate, steps=num_samples)
    axis.plot(time_axis, waveform.data, linewidth=0.1)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.set_xlim(float(time_axis[0]), float(time_axis[-1]))


def plot_spectrogram(waveform: AudioWaveform, axis: Axes) -> None:
    resample = T.Resample(waveform.sample_rate, 16_000)
    waveform_to_spectrogram = T.Spectrogram(win_length=4096, n_fft=4096, hop_length=2048)
    db_transform = T.AmplitudeToDB(stype="power", top_db=80)
    spectrogram = db_transform(waveform_to_spectrogram(resample(waveform.data)))
    axis.imshow(
        spectrogram, aspect="auto", origin="lower", extent=(0, waveform.duration, 0, waveform.sample_rate / 2000)
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Frequency (kHz)")
