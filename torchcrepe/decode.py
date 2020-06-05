import librosa
import numpy as np
import torch

from .core import bins_to_cents, bins_to_frequency, cents_to_frequency


###############################################################################
# Probability sequence decoding methods
###############################################################################


def argmax(logits):
    """Sample observations by taking the argmax"""
    return bins_to_frequency(logits.argmax(dim=1))


def weighted_argmax(logits):
    """Sample observations using weighted sum near the argmax"""
    # Convert to probabilities
    with torch.no_grad():
        probs = torch.sigmoid(logits)
    
    # Find center of analysis window
    center = torch.argmax(probs, dim=1)
    
    # Find bounds of analysis window
    start = torch.max(torch.tensor(0), center - 4)
    end = torch.min(torch.tensor(probs.size(2)), center + 5)
    
    # Mask out everything outside of window
    for batch in range(probs.size(0)):
        for time in range(probs.size(2)):
            probs[batch, :start[batch, time], time] = 0.
            probs[batch, end[batch, time]:, time] = 0.

    # Construct weights
    if not hasattr(weighted_argmax, 'weights'):
        weights = bins_to_cents(torch.arange(360))
        weighted_argmax.weights = weights[None, :, None]
    
    # Ensure devices are the same (no-op if they are)
    weighted_argmax.weights = weighted_argmax.weights.to(probs.device)
    
    # Apply weights
    cents = (weighted_argmax.weights * probs).sum(dim=1) / probs.sum(dim=1)
    
    # Convert to frequency in Hz
    return cents_to_frequency(cents)


def viterbi(logits):
    """Sample observations using viterbi decoding"""
    # Create viterbi transition matrix
    if not hasattr(viterbi, 'transition'):
        xx, yy = np.meshgrid(range(360), range(360))
        transition = np.maximum(12 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        viterbi.transition = transition
    
    # Normalize logits
    with torch.no_grad():
        probs = torch.nn.functional.softmax(logits, dim=1)
        
    # Convert to numpy
    sequences = probs.cpu().numpy()
    
    # Perform viterbi decoding
    bins = [librosa.sequence.viterbi(sequence, viterbi.transition)
                    for sequence in sequences]
    
    # Convert to pytorch
    bins = torch.tensor(bins, device=probs.device)
    
    # Convert to frequency in Hz
    return bins_to_frequency(bins)
