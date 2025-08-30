from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Final

import librosa
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from hs_hackathon_drone_acoustics.base import AudioWaveform

logger: Final = logging.getLogger(__name__)


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, waveform: AudioWaveform) -> NDArray[np.float64]: ...


class EnergyFeatureExtractor(FeatureExtractor):
    def extract(self, waveform: AudioWaveform) -> NDArray[np.float64]:
        energy = np.linalg.vector_norm(waveform.data, ord=2)
        return np.array([energy])


class MFCCFeatureExtractor(FeatureExtractor):
    def __init__(self, n_mfcc: int = 20):
        self._n_mfcc = n_mfcc

    def extract(self, waveform: AudioWaveform) -> NDArray[np.float64]:
        mfccs = librosa.feature.mfcc(y=waveform.data.numpy(), sr=waveform.sample_rate, n_mfcc=self._n_mfcc)
        ensemble_mfcc: NDArray[np.float64] = np.mean(mfccs, axis=1)
        return ensemble_mfcc


class FeatureExtractorPipeline(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    def __init__(self, *feature_extractors: FeatureExtractor) -> None:
        super().__init__()
        self._feature_extractors = feature_extractors

    def fit(self, *args: Any, **kwargs: Any) -> FeatureExtractorPipeline:
        return self

    def transform(self, waveforms: list[AudioWaveform]) -> NDArray[np.float64]:
        if not isinstance(waveforms, list):
            raise TypeError("Input x must be a list of AudioWaveform objects.")
        all_features = []
        for waveform in waveforms:
            waveform_features = []
            for feature_extractor in self._feature_extractors:
                waveform_features.append(feature_extractor.extract(waveform))
            all_features.append(np.concatenate(waveform_features))
        return np.stack(all_features)
