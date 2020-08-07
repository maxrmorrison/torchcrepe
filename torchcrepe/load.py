import torch
from scipy.io import wavfile


def audio(filename):
    """Load audio from disk"""
    sample_rate, audio = wavfile.read(filename)

    return torch.tensor(audio)[None], sample_rate
