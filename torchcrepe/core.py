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


def predict(audio, sample_rate, hop_length, fmin=0., fmax=7180., viterbi=False):
    """Performs pitch estimation
    
    Arguments
        audio (torch.tensor [shape=(time,)])
            The audio signals
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
        viterbi (bool)
            Whether to use viterbi decoding
    
    Returns
        pitch (torch.tensor [shape=(time / hop_length,)])
    """
    # Preprocess audio
    frames = preprocess(audio, sample_rate, hop_length)
    
    # Infer independent probabilities for each pitch bin
    probabilities = infer(frames)
    
    # Convert probabilities to F0 and harmonicity
    return postprocess(probabilities, viterbi)


###############################################################################
# Components for step-by-step prediction
###############################################################################


def infer(batch):
    """Forward pass through the model
    
    Arguments
        frames (torch.tensor [shape=(1024, time / hop_length)])
    
    Returns 
        probabilities (torch.tensor [shape=(360, time / hop_length)])
    """
    # Load the model if necessary
    if not hasattr(infer, 'model'):
        infer.model = Crepe('torchcrepe/weights.npy')
    
    # Move model to correct device (no-op if devices are the same)
    infer.model = infer.model.to(batch.device)
    
    # Apply model
    return infer.model(batch.t()).t()


def postprocess(probabilities, viterbi=False):
    """Convert model output to F0 and harmonicity
    
    Arguments
        probabilities (torch.tensor [shape=(360, time / hop_length)])
            The probabilities for each pitch bin inferred by the network
        viterbi (bool)
            Whether to use viterbi decoding
    """
    # Sampling is non-differentiable, so remove from graph
    probabilities = probabilities.detach()
    
    # Use maximum probability over pitch bins as harmonicity
    harmonicity = probabilities.max(dim=0).values
    
    # Perform argmax or viterbi sampling
    if viterbi:
        bins = viterbi_decode(probabilities)
    else:
        bins = probabilities.argmax(dim=0)
        
    # Convert to frequencies in Hz
    return bins_to_frequency(bins), harmonicity
        
    
def preprocess(audio, sample_rate, hop_length):
    """Convert audio to model input
    
    Arguments
        audio (torch.tensor [shape=(time,)]) - The audio signals
        sample_rate (int) - The sampling rate in Hz
        hop_length (int) - The hop_length in samples
    
    Returns
        frames (torch.tensor [shape=(1024, time / hop_length)])
    """
    # Add dummy dimension
    audio = audio.unsqueeze(0)
    
    # Resample
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(
            sample_rate, 16000)(audio)
        hop_length = int(hop_length * 16000 / sample_rate)
    
    # Pad
    audio = torch.nn.functional.pad(audio, (512, 512))
        
    # Frame
    frames = torch.nn.functional.unfold(
        audio[None, None, :, :],
        kernel_size=(1, 1024),
        stride=(1, hop_length)).squeeze(0)
    
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


def viterbi_decode(probabilities):
    """Sample observations using viterbi decoding"""
    # Create viterbi transition matrix
    if not hasattr(viterbi_decode, 'transition'):
        xx, yy = np.meshgrid(range(360), range(360))
        transition = np.maximum(12 - abs(xx - yy), 0)
        transition = transition / np.sum(transition, axis=1)[:, None]
        viterbi_decode.transition = transition
    
    # Normalize probabilities
    with torch.no_grad():
        posterior = torch.nn.functional.softmax(probabilities, dim=0)
    
    # Convert to numpy
    posterior = posterior.cpu().numpy()
    
    # Perform viterbi decoding
    # TODO - this is currently producing all zeros on speech commands data
    observations = librosa.sequence.viterbi(
        posterior, viterbi_decode.transition)
    
    # Convert to pytorch
    return torch.tensor(observations, device=probabilities.device)
