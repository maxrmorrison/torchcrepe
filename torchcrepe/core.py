import os

import numpy as np
import torch

import torchcrepe


__all__ = ['CENTS_PER_BIN',
           'MAX_FMAX',
           'PITCH_BINS',
           'SAMPLE_RATE',
           'WINDOW_SIZE',
           'UNVOICED',
           'embed',
           'infer',
           'predict',
           'preprocess',
           'postprocess',
           'resample']


###############################################################################
# Constants
###############################################################################


CENTS_PER_BIN = 20  # cents
MAX_FMAX = 2006.  # hz
PITCH_BINS = 360
SAMPLE_RATE = 16000  # hz
WINDOW_SIZE = 1024  # samples
UNVOICED = np.nan


###############################################################################
# Crepe pitch prediction
###############################################################################


def embed(audio, sample_rate, hop_length, model='full'):
    """Embeds audio to the output of CREPE's fifth maxpool layer
    
    Arguments
        audio (torch.tensor [shape=(1, time)])
            The audio signals
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
        model (string)
            The model capacity. One of 'full' or 'tiny'.
            
    Returns
        embedding (torch.tensor [shape=(1, time / hop_length, 32, 8)])
    """
    # Preprocess audio
    frames = preprocess(audio, sample_rate, hop_length)
    
    # Infer pitch embeddings
    embeddings = infer(frames, model, embed=True)
    
    # shape=(batch, time / hop_length, 32, embedding_size)
    return embeddings.reshape(audio.size(0), frames.size(0), 32, -1)


def predict(audio,
            sample_rate,
            hop_length,
            fmin=0.,
            fmax=MAX_FMAX,
            model='full',
            decoder=torchcrepe.decode.viterbi,
            return_harmonicity=False):
    """Performs pitch estimation
    
    Arguments
        audio (torch.tensor [shape=(1, time)])
            The audio signal
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        model (string)
            The model capacity. One of 'full' or 'tiny'.
        decoder (function)
            The decoder to use. See decode.py for decoders.
        return_harmonicity (bool)
            Whether to also return the network confidence

    Returns
        pitch (torch.tensor [shape=(1, time / hop_length)])
        (Optional) harmonicity(torch.tensor [shape=(1, time / hop_length)])
    """
    # Preprocess audio
    frames = preprocess(audio, sample_rate, hop_length)
    
    # Infer independent probabilities for each pitch bin
    probabilities = infer(frames, model)
    
    # shape=(batch, 360, time / hop_length)
    probabilities = probabilities.reshape(
        audio.size(0), -1, PITCH_BINS).transpose(1, 2)
    
    # Convert probabilities to F0 and harmonicity
    return postprocess(probabilities, fmin, fmax, decoder, return_harmonicity)


###############################################################################
# Components for step-by-step prediction
###############################################################################


def infer(frames, model='full', embed=False):
    """Forward pass through the model
    
    Arguments
        frames (torch.tensor [shape=(time / hop_length, 1024)])
            The network input
        model (string)
            The model capacity. One of 'full' or 'tiny'.
        embed (bool)
            Whether to stop inference at the intermediate embedding layer
    
    Returns 
        logits (torch.tensor [shape=(time / hop_length, 360)]) OR
        embedding (torch.tensor [shape=(time / hop_length, embedding_size)])
    """
    # Load the model if necessary
    if not hasattr(infer, 'model') or not hasattr(infer, 'capacity') or \
        (hasattr(infer, 'capacity') and infer.capacity != model):
        infer.model = torchcrepe.Crepe(model)
        
        # Load weights
        file = os.path.join(
            os.path.dirname(__file__), 'assets', f'{model}.pth')
        infer.model.load_state_dict(torch.load(file))
        
        infer.model.eval()
        infer.capacity = model
        
    # Move model to correct device (no-op if devices are the same)
    infer.model = infer.model.to(frames.device)
    
    # Apply model
    return infer.model(frames, embed=embed)


def postprocess(probabilities,
                fmin=0.,
                fmax=MAX_FMAX,
                decoder=torchcrepe.decode.viterbi,
                return_harmonicity=False):
    """Convert model output to F0 and harmonicity
    
    Arguments
        probabilities (torch.tensor [shape=(1, 360, time / hop_length)])
            The probabilities for each pitch bin inferred by the network
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        viterbi (bool)
            Whether to use viterbi decoding
        return_harmonicity (bool)
            Whether to also return the network confidence
            
    Returns
        pitch (torch.tensor [shape=(1, time / hop_length)])
        harmonicity (torch.tensor [shape=(1, time / hop_length)])
    """
    # Sampling is non-differentiable, so remove from graph
    probabilities = probabilities.detach()
    
    # Convert frequency range to pitch bin range
    minidx = torchcrepe.convert.frequency_to_bins(torch.tensor(fmin))
    maxidx = torchcrepe.convert.frequency_to_bins(torch.tensor(fmax),
                                                  torch.ceil)
    
    # Remove frequencies outside of allowable range
    probabilities[:, :minidx] = -float('inf')
    probabilities[:, maxidx:] = -float('inf')
    
    # Perform argmax or viterbi sampling
    bins, pitch = decoder(probabilities)
    
    if not return_harmonicity:
        return pitch
    
    # Compute harmonicity from probabilities and decoded pitch bins
    return pitch, harmonicity(probabilities, bins)
        
    
def preprocess(audio, sample_rate, hop_length):
    """Convert audio to model input
    
    Arguments
        audio (torch.tensor [shape=(1, time)]) - The audio signals
        sample_rate (int) - The sampling rate in Hz
        hop_length (int) - The hop_length in samples
    
    Returns
        frames (torch.tensor [shape=(time / hop_length, 1024)])
    """
    # Resample
    if sample_rate != SAMPLE_RATE:
        audio = resample(audio, sample_rate)
        hop_length = int(hop_length * SAMPLE_RATE / sample_rate)
        
    # Pad
    audio = torch.nn.functional.pad(audio,
                                    (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    
    # Chunk
    frames = torch.nn.functional.unfold(
        audio[:, None, None, :],
        kernel_size=(1, WINDOW_SIZE),
        stride=(1, hop_length))
    
    # shape=(batch * time / hop_length, 1024)
    frames = frames.transpose(1, 2).reshape(-1, WINDOW_SIZE)
    
    # Normalize
    frames -= frames.mean(dim=1, keepdim=True)
    
    # Note: during silent frames, this produces very large numbers. But this
    # seems to be what crepe expects.
    frames /= frames.std(dim=1, keepdim=True)
    
    return frames


###############################################################################
# Utilities
###############################################################################


def harmonicity(probabilities, bins):
    """Computes the harmonicity from the network output and pitch bins"""
    # shape=(batch * time / hop_length, 360)
    probs_stacked = probabilities.transpose(1, 2).reshape(-1, PITCH_BINS)
    
    # shape=(batch * time / hop_length, 1)
    bins_stacked = bins.reshape(-1, 1)
    
    # Use maximum logit over pitch bins as harmonicity
    harmonicity = probs_stacked.gather(1, bins_stacked)
    
    # shape=(batch, time / hop_length)
    return harmonicity.reshape(probabilities.size(0), probabilities.size(2))


def resample(audio, sample_rate):
    """Resample audio"""
    import resampy
    
    # Store device for later placement
    device = audio.device
    
    # Convert to numpy
    audio = audio.detach().cpu().numpy().squeeze(0)
    
    # Resample
    # We have to use resampy if we want numbers to match Crepe
    audio = resampy.resample(audio, sample_rate, SAMPLE_RATE)
    
    # Convert to pytorch
    return torch.tensor(audio, device=device).unsqueeze(0)
