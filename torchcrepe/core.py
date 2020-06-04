import os

import librosa
import numpy as np
import torch
import torchaudio

from .model import Crepe


__all__ = ['bins_to_cents',
           'bins_to_frequency',
           'infer',
           'predict',
           'preprocess',
           'postprocess',
           'viterbi_decode']


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


def predict(audio, sample_rate, hop_length, fmin=0., fmax=7180., viterbi=False):
    """Performs pitch estimation
    
    Arguments
        audio (torch.tensor [shape=(batch, time)])
            The audio signals
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
        viterbi (bool)
            Whether to use viterbi decoding
    
    Returns
        pitch (torch.tensor [shape=(batch, time / hop_length)])
        harmonicity (torch.tensor [shape=(batch, time / hop_length)])
    """
    # Preprocess audio
    frames = preprocess(audio, sample_rate, hop_length)
    
    # Infer independent probabilities for each pitch bin
    probs = infer(frames)
    
    # shape=(batch, 360, time / hop_length)
    probs = probs.reshape(audio.size(0), -1, 360).transpose(1, 2)
    
    # Convert probabilities to F0 and harmonicity
    return postprocess(probs, viterbi)


###############################################################################
# Components for step-by-step prediction
###############################################################################


def infer(batch, embed=False):
    """Forward pass through the model
    
    Arguments
        frames (torch.tensor [shape=(batch * time / hop_length, 1024)])
    
    Returns 
        probabilities (torch.tensor [shape=(batch * time / hop_length, 360)])
    """
    # Load the model if necessary
    if not hasattr(infer, 'model'):
        directory = os.path.dirname(os.path.realpath(__file__))
        infer.model = Crepe(os.path.join(directory, 'weights.npy'))
    
    # Move model to correct device (no-op if devices are the same)
    infer.model = infer.model.to(batch.device)
    
    # Apply model
    return infer.model(batch, embed=embed)


def postprocess(probs, viterbi=False):
    """Convert model output to F0 and harmonicity
    
    Arguments
        probs (torch.tensor [shape=(batch, 360, time / hop_length)])
            The probabilities for each pitch bin inferred by the network
        viterbi (bool)
            Whether to use viterbi decoding
            
    Returns
        pitch (torch.tensor [shape=(batch, time / hop_length)])
        harmonicity (torch.tensor [shape=(batch, time / hop_length)])
    """
    # Sampling is non-differentiable, so remove from graph
    probs = probs.detach()
    
    # Use maximum probability over pitch bins as harmonicity
    harmonicity = probs.max(dim=1).values
    
    # Perform argmax or viterbi sampling
    bins = viterbi_decode(probs) if viterbi else probs.argmax(dim=1)
        
    # Convert to frequencies in Hz
    return bins_to_frequency(bins), harmonicity
        
    
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
    return 10 * 2 ** (bins_to_cents(bins) / 1200)


def viterbi_decode(probs):
    """Sample observations using viterbi decoding"""
    # Create viterbi transition matrix
    if not hasattr(viterbi_decode, 'transition'):
        xx, yy = np.meshgrid(range(360), range(360))
        transition = np.maximum(12 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        viterbi_decode.transition = transition
    
    # Normalize probabilities
    with torch.no_grad():
        posterior = torch.nn.functional.softmax(probs, dim=0)
    
    # Convert to numpy
    posterior = posterior.cpu().numpy()
    
    # Perform viterbi decoding
    # TODO - this is currently producing all zeros on speech commands data
    observations = librosa.sequence.viterbi(
        posterior, viterbi_decode.transition)
    
    # Convert to pytorch
    return torch.tensor(observations, device=probs.device)
