import numpy as np
import torch
from scipy.io import wavfile


def audio(filename):
    """Load audio from disk"""
    sample_rate, audio = wavfile.read(filename)

    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max

    return torch.tensor(audio)[None], sample_rate
