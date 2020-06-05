import os

import librosa
import numpy as np
import torch
import torchaudio

from . import convert
from . import decode
from .model import Crepe


__all__ = ['embed',
           'infer',
           'predict',
           'preprocess',
           'postprocess',
           'threshold']


###############################################################################
# Crepe pitch prediction
###############################################################################


def embed(audio, sample_rate, hop_length):
    """Embeds audio to the output of CREPE's fifth maxpool layer
    
    Arguments
        audio (torch.tensor [shape=(batch, time)])
            The audio signals
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
            
    Returns
        embedding (torch.tensor [shape=(batch, time / hop_length, 32, 8)])
    """
    # Preprocess audio
    frames = preprocess(audio, sample_rate, hop_length)
    
    # Infer pitch embeddings
    embeddings = infer(frames, embed=True)
    
    # shape=(batch, time / hop_length, 32, 8)
    return embeddings.reshape(audio.size(0), -1, 32, 8)


def predict(audio,
            sample_rate,
            hop_length,
            fmin=0.,
            fmax=7180.,
            decoder=decode.viterbi):
    """Performs pitch estimation
    
    Arguments
        audio (torch.tensor [shape=(batch, time)])
            The audio signals
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        decoder (function)
            The decoder to use. See decode.py for decoders.
    
    Returns
        pitch (torch.tensor [shape=(batch, time / hop_length)])
        harmonicity (torch.tensor [shape=(batch, time / hop_length)])
    """
    # Preprocess audio
    frames = preprocess(audio, sample_rate, hop_length)
    
    # Infer independent probabilities for each pitch bin
    logits = infer(frames)
    
    # shape=(batch, 360, time / hop_length)
    logits = logits.reshape(audio.size(0), -1, 360).transpose(1, 2)
    
    # Convert probabilities to F0 and harmonicity
    return postprocess(logits, fmin, fmax, decoder)


###############################################################################
# Components for step-by-step prediction
###############################################################################


def infer(batch, embed=False):
    """Forward pass through the model
    
    Arguments
        frames (torch.tensor [shape=(batch * time / hop_length, 1024)])
    
    Returns 
        logits (torch.tensor [shape=(batch * time / hop_length, 360)])
    """
    # Load the model if necessary
    if not hasattr(infer, 'model'):
        directory = os.path.dirname(os.path.realpath(__file__))
        infer.model = Crepe(os.path.join(directory, 'weights.npy'))
    
    # Move model to correct device (no-op if devices are the same)
    infer.model = infer.model.to(batch.device)
    
    # Apply model
    return infer.model(batch, embed=embed)


def postprocess(logits, fmin=0., fmax=2006., decoder=decode.viterbi):
    """Convert model output to F0 and harmonicity
    
    Arguments
        logits (torch.tensor [shape=(batch, 360, time / hop_length)])
            The logits for each pitch bin inferred by the network
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        viterbi (bool)
            Whether to use viterbi decoding
            
    Returns
        pitch (torch.tensor [shape=(batch, time / hop_length)])
        harmonicity (torch.tensor [shape=(batch, time / hop_length)])
    """
    # Sampling is non-differentiable, so remove from graph
    logits = logits.detach()
    
    # Convert frequency range to pitch bin range
    minidx = convert.frequency_to_bins(torch.tensor(fmin))
    maxidx = convert.frequency_to_bins(torch.tensor(fmax), torch.ceil)
    
    # Remove frequencies outside of allowable range
    logits[:, :minidx] = -float('inf')
    logits[:, maxidx:] = -float('inf')
    
    # Perform argmax or viterbi sampling
    bins, pitch = decoder(logits)
    
    # Compute harmonicity from logits and decoded pitch bins
    return pitch, harmonicity(logits, bins)
        
    
def preprocess(audio, sample_rate, hop_length):
    """Convert audio to model input
    
    Arguments
        audio (torch.tensor [shape=(batch, time)]) - The audio signals
        sample_rate (int) - The sampling rate in Hz
        hop_length (int) - The hop_length in samples
    
    Returns
        frames (torch.tensor [shape=(batch * time / hop_length, 1024)])
    """
    # Resample
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(
            sample_rate, 16000)(audio)
        hop_length = int(hop_length * 16000 / sample_rate)
    
    # Pad
    audio = torch.nn.functional.pad(audio, (512, 512))
        
    # Frame
    frames = torch.nn.functional.unfold(
        audio[:, None, None, :],
        kernel_size=(1, 1024),
        stride=(1, hop_length))
    
    # shape=(batch * time / hop_length, 1024)
    frames = frames.transpose(1, 2).reshape(-1, 1024)
    
    # Normalize
    frames -= frames.mean(dim=0, keepdim=True)
    std = frames.std(dim=0, keepdim=True)
    not_silent = std.squeeze() > 1e-8
    frames[:, not_silent] /= std[:, not_silent]
    
    return frames


###############################################################################
# Utilities
###############################################################################


def harmonicity(logits, bins):
    """Computes the harmonicity from the network output and pitch bins"""
    # Normalize logits
    probs = torch.sigmoid(logits)
    
    # shape=(batch * time / hop_length, 360)
    probs_stacked = probs.transpose(1, 2).reshape(-1, 360)
    
    # shape=(batch * time / hop_length,, 1)
    bins_stacked = bins.reshape(-1, 1)
    
    # Use maximum logit over pitch bins as harmonicity
    harmonicity = probs_stacked.gather(1, bins_stacked)
    
    # shape=(batch, time / hop_length)
    return harmonicity.reshape(probs.size(0), probs.size(2))
    
    
def threshold(pitch, harmonicity, value):
    """Mask inharmonic pitch values with nans
    
    Arguments
        pitch (torch.tensor [shape=(batch, time)])
            The pitch contours
        harmonicity (torch.tensor [shape=(batch, time)])
            The harmonicity confidence values
        value (float)
            The threshold value
    
    Returns
        thresholded (torch.tensor [shape=(batch, time)])
    """
    pitch[harmonicity < value] = np.nan
    return pitch
