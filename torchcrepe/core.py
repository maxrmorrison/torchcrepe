import os

import librosa
import numpy as np
import torch
import torchaudio

from . import decode
from .model import Crepe


__all__ = ['bins_to_cents',
           'bins_to_frequency',
           'cents_to_bins',
           'frequency_to_bins',
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
        embedding (torch.tensor [shape=(batch, channels, time / hop_length)])
    """
    # Preprocess audio
    frames = preprocess(audio, sample_rate, hop_length)
    
    # Infer pitch embeddings
    return infer(frames, embed=true)


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
    minidx = frequency_to_bins(torch.tensor(fmin))
    maxidx = frequency_to_bins(torch.tensor(fmax), torch.ceil)
    
    # Remove frequencies outside of allowable range
    logits[:, :minidx] = -float('inf')
    logits[:, maxidx:] = -float('inf')
    
    # Normalize logits
    probs = torch.sigmoid(logits)
    
    # Use maximum logit over pitch bins as harmonicity
    harmonicity = probs.max(dim=1).values
    
    # Perform argmax or viterbi sampling
    pitch = decoder(logits)
        
    # Convert to frequencies in Hz
    return pitch, harmonicity
        
    
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


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    return 20. * bins + 1997.3794084376191


def bins_to_frequency(bins):
    """Converts pitch bins to frequency in Hz"""
    return cents_to_frequency(bins_to_cents(bins))


def cents_to_bins(cents, quantize_fn=torch.floor):
    """Converts cents to pitch bins"""
    return quantize_fn((cents - 1997.3794084376191) / 20.).int()


def cents_to_frequency(cents):
    """Converts cents to frequency in Hz"""
    return 10 * 2 ** (cents / 1200)


def frequency_to_bins(frequency, quantize_fn=torch.floor):
    """Convert frequency in Hz to pitch bins"""
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)


def frequency_to_cents(frequency):
    """Convert frequency in Hz to cents"""
    return 1200 * torch.log2(frequency / 10.)


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
