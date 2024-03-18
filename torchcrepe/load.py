import os

import numpy as np
import torch
import torchcrepe
from scipy.io import wavfile


def audio(filename):
    """Load audio from disk"""
    sample_rate, audio = wavfile.read(filename)

    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max

    # PyTorch is not compatible with non-writeable arrays, so we make a copy
    return torch.tensor(np.copy(audio))[None], sample_rate


def model(device, capacity="full", dtype=torch.float32, compile=False):
    """Preloads model from disk"""
    # Bind model and capacity
    torchcrepe.infer.capacity = capacity
    torchcrepe.infer.dtype = dtype
    torchcrepe.infer.model = torchcrepe.Crepe(capacity, dtype=dtype)

    # Load weights
    file = os.path.join(os.path.dirname(__file__), "assets", f"{capacity}.pth")
    torchcrepe.infer.model.load_state_dict(torch.load(file, map_location=device))

    # Place on device
    torchcrepe.infer.model = torchcrepe.infer.model.to(torch.device(device))

    # Eval mode
    torchcrepe.infer.model.eval()

    # compile model
    torchcrepe.infer.compile = compile
    if compile:
        torchcrepe.infer.model = torch.compile(torchcrepe.infer.model)
