from functools import lru_cache

import librosa
import numpy as np
import torch

import torchcrepe


###############################################################################
# Probability sequence decoding methods
###############################################################################


_BINS_TO_CENTS_OFFSET = 1997.3794084376191


def _bins_to_frequency(bins, dither=True):
    cents = bins.to(torch.float32) * torchcrepe.CENTS_PER_BIN + _BINS_TO_CENTS_OFFSET
    if dither:
        cents = torchcrepe.convert.dither(cents)
    return torchcrepe.convert.cents_to_frequency(cents)


def argmax(logits, *, dither=True):
    """Sample observations by taking the argmax"""
    bins = logits.argmax(dim=1)

    # Convert to frequency in Hz
    return bins, _bins_to_frequency(bins, dither=dither)


def weighted_argmax(logits, *, dither=True):
    """Sample observations using weighted sum near the argmax"""
    logits = logits.clone()

    # Find center of analysis window
    bins = logits.argmax(dim=1)

    # Find bounds of analysis window
    start = torch.max(torch.tensor(0, device=logits.device), bins - 4)
    end = torch.min(torch.tensor(logits.size(1), device=logits.device), bins + 5)

    # Mask out everything outside of window
    for batch in range(logits.size(0)):
        for time in range(logits.size(2)):
            logits[batch, :start[batch, time], time] = -float('inf')
            logits[batch, end[batch, time]:, time] = -float('inf')

    # Construct weights
    if not hasattr(weighted_argmax, 'weights'):
        weights = torchcrepe.convert.bins_to_cents(torch.arange(360))
        weighted_argmax.weights = weights[None, :, None]

    # Ensure devices are the same (no-op if they are)
    weighted_argmax.weights = weighted_argmax.weights.to(logits.device)

    # Convert to probabilities
    with torch.no_grad():
        probs = torch.sigmoid(logits)

    # Apply weights
    cents = (weighted_argmax.weights * probs).sum(dim=1) / probs.sum(dim=1)

    # Convert to frequency in Hz
    if dither:
        cents = torchcrepe.convert.dither(cents)
    return bins, torchcrepe.convert.cents_to_frequency(cents)


@lru_cache(maxsize=1)
def _legacy_transition():
    xx, yy = np.meshgrid(np.arange(torchcrepe.PITCH_BINS),
                         np.arange(torchcrepe.PITCH_BINS))
    transition = np.maximum(12 - np.abs(xx - yy), 0).astype(np.float32)
    transition = transition / transition.sum(axis=1, keepdims=True)
    return transition


@lru_cache(maxsize=1)
def _transition_banded():
    transition = _legacy_transition()
    valid = transition > 0
    width = int(valid.sum(axis=0).max())
    source_idx = np.zeros((torchcrepe.PITCH_BINS, width), dtype=np.int64)
    log_weights = np.full((torchcrepe.PITCH_BINS, width),
                          -np.inf,
                          dtype=np.float32)
    for target in range(torchcrepe.PITCH_BINS):
        sources = np.flatnonzero(valid[:, target]).astype(np.int64)
        count = sources.size
        source_idx[target, :count] = sources
        log_weights[target, :count] = np.log(transition[sources, target])
    return source_idx, log_weights


def _log_observations_from_probabilities(probabilities):
    observations = probabilities.detach().cpu().numpy().astype(np.float32, copy=False)
    # log_softmax(x) = x - logsumexp(x); the per-frame constant does not affect
    # the Viterbi path, so decode from shifted values directly.
    observations = observations - np.max(observations, axis=1, keepdims=True)
    return observations


def _viterbi_bins_numpy(log_obs, source_idx, log_trans):
    batches, states, timesteps = log_obs.shape
    bins = np.empty((batches, timesteps), dtype=np.int64)
    backpointers = np.empty((timesteps, states), dtype=np.int64)
    state_idx = np.arange(states, dtype=np.int64)

    for batch in range(batches):
        score = log_obs[batch, :, 0].copy()
        backpointers[0] = state_idx
        for t in range(1, timesteps):
            obs_t = log_obs[batch, :, t]
            candidates = score[source_idx] + log_trans
            best_offsets = np.argmax(candidates, axis=1)
            backpointers[t] = source_idx[state_idx, best_offsets]
            score = candidates[state_idx, best_offsets] + obs_t
        bins[batch, -1] = int(np.argmax(score))
        for t in range(timesteps - 1, 0, -1):
            bins[batch, t - 1] = backpointers[t, bins[batch, t]]
    return bins


def viterbi_legacy(probabilities, *, dither=True):
    """Sample observations using the legacy librosa Viterbi path"""
    with torch.no_grad():
        sequences = torch.nn.functional.softmax(probabilities, dim=1).cpu().numpy()

    bins = np.array([
        librosa.sequence.viterbi(sequence, _legacy_transition()).astype(np.int64)
        for sequence in sequences])
    bins = torch.tensor(bins, device=probabilities.device)
    return bins, _bins_to_frequency(bins, dither=dither)


def viterbi_banded_fast(probabilities, *, dither=True):
    """Sample observations using an exact banded Viterbi implementation"""
    source_idx, log_trans = _transition_banded()
    log_obs = _log_observations_from_probabilities(probabilities)
    bins = _viterbi_bins_numpy(log_obs, source_idx, log_trans)
    bins = torch.tensor(bins, device=probabilities.device)
    return bins, _bins_to_frequency(bins, dither=dither)


def viterbi(probabilities, *, dither=True):
    """Sample observations using the legacy Viterbi decoder"""
    return viterbi_legacy(probabilities, dither=dither)


DECODERS = {
    'argmax': argmax,
    'weighted_argmax': weighted_argmax,
    'viterbi': viterbi,
    'viterbi_legacy': viterbi_legacy,
    'viterbi_banded_fast': viterbi_banded_fast,
}


def get_decoder(name):
    try:
        return DECODERS[name]
    except KeyError as exc:
        raise ValueError(f'Unknown decoder: {name}') from exc
