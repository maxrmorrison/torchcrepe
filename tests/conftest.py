import os

import numpy as np
import pytest
import torch

import torchcrepe


###############################################################################
# Testing fixtures
###############################################################################


@pytest.fixture(scope='session')
def activation_full():
    """Retrieve the original crepe activation on the test audio"""
    return np.load(path('activation-full.npy'))


@pytest.fixture(scope='session')
def activation_tiny():
    """Retrieve the original crepe activation on the test audio"""
    return np.load(path('activation-tiny.npy'))


@pytest.fixture(scope='session')
def audio():
    """Retrieve the test audio"""
    audio, sample_rate = torchcrepe.load.audio(path('test.wav'))
    if sample_rate != torchcrepe.SAMPLE_RATE:
        audio = torchcrepe.resample(audio, sample_rate)
    return audio


@pytest.fixture(scope='session')
def frames():
    """Retrieve the preprocessed frames for inference

    Note: the reason we load frames from disk rather than compute ourselves
    is that the normalizing process in the preprocessing isn't numerically
    stable. Therefore, we use the exact same preprocessed features that were
    passed through crepe to retrieve the activations--thus bypassing the
    preprocessing step.
    """
    return torch.tensor(np.load(path('frames-crepe.npy')))


###############################################################################
# Utilities
###############################################################################


def path(file):
    """Retrieve the path to the test file"""
    return os.path.join(os.path.dirname(__file__), 'assets', file)
