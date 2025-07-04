import torch
import torchaudio

import torchcrepe
from huggingface_hub import hf_hub_download

def audio(filename):
    """Load audio from disk"""
    return torchaudio.load(filename)


def model(device, capacity='full'):
    """Preloads model from disk"""
    # Bind model and capacity
    torchcrepe.infer.capacity = capacity
    torchcrepe.infer.model = torchcrepe.Crepe(capacity)

    # Load weights
    file = hf_hub_download("shethjenil/Audio2Midi_Models",f"crepe_{capacity}.pt")
    torchcrepe.infer.model.load_state_dict(torch.load(file, map_location=device, weights_only=True))

    # Place on device
    torchcrepe.infer.model = torchcrepe.infer.model.to(torch.device(device))

    # Eval mode
    torchcrepe.infer.model.eval()
