import os

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile

import torchcrepe


def audio(filename):
    """Load audio from disk"""
    return torchaudio.load(filename)


def model(device, capacity='full'):
    """Preloads model from disk"""
    # Bind model and capacity
    torchcrepe.infer.capacity = capacity
    torchcrepe.infer.model = torchcrepe.Crepe(capacity)

    # Load weights
    file = os.path.join(os.path.dirname(__file__), 'assets', f'{capacity}.pth')
    torchcrepe.infer.model.load_state_dict(
        torch.load(file, map_location=device))

    # Place on device
    torchcrepe.infer.model = torchcrepe.infer.model.to(torch.device(device))

    # Eval mode
    torchcrepe.infer.model.eval()
