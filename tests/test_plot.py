from matplotlib import pyplot as plt

from hs_hackathon_drone_acoustics import EXAMPLES_DIR
from hs_hackathon_drone_acoustics.base import AudioWaveform
from hs_hackathon_drone_acoustics.plot import plot_spectrogram, plot_waveform


def test__plot_waveform__okay() -> None:
    for file in ["BACKGROUND_001_L.wav", "DRONE_001_L.wav", "HELICOPTER_001_L_0.wav"]:
        path = EXAMPLES_DIR / file
        waveform = AudioWaveform.load(path)
        fig, axis = plt.subplots(nrows=1, ncols=1)
        plot_waveform(waveform, axis)
        fig.suptitle(f"{path.stem} Waveform")
        plt.tight_layout()


def test__plot_spectrogram__okay() -> None:
    for file in ["BACKGROUND_001_L.wav", "DRONE_001_L.wav", "HELICOPTER_001_L_0.wav"]:
        path = EXAMPLES_DIR / file
        waveform = AudioWaveform.load(path)
        fig, axis = plt.subplots(nrows=1, ncols=1)
        plot_spectrogram(waveform, axis)
        fig.suptitle(f"{path.stem} Spectrogram")
        plt.tight_layout()
