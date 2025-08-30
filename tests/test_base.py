import torch

from hs_hackathon_drone_acoustics import EXAMPLES_DIR
from hs_hackathon_drone_acoustics.base import AudioWaveform


def test__AudioWaveform__load() -> None:
    waveform = AudioWaveform.load(EXAMPLES_DIR / "BACKGROUND_001_L.wav")
    assert waveform.sample_rate == 44_100


def test__AudioWaveform__duration() -> None:
    waveform = AudioWaveform(torch.rand(10 * 100), sample_rate=100)
    assert waveform.duration == 10
