import numpy as np
import torchcrepe


###############################################################################
# Test core.py
###############################################################################


def test_embed_tiny(audio):
    """Tests that the embedding is the expected size"""
    embedding = torchcrepe.embed(audio, torchcrepe.SAMPLE_RATE, 160, 'tiny')
    assert embedding.size() == (1, 1001, 32, 8)


def test_embed_full(audio):
    """Tests that the embedding is the expected size"""
    embedding = torchcrepe.embed(audio, torchcrepe.SAMPLE_RATE, 160, 'full')
    assert embedding.size() == (1, 1001, 32, 64)


def test_infer_tiny(frames, activation_tiny):
    """Test that inference is the same as the original crepe"""
    activation = torchcrepe.infer(frames, 'tiny').detach()
    diff = np.abs(activation - activation_tiny)
    assert diff.max() < 1e-5 and diff.mean() < 1e-7


def test_infer_full(frames, activation_full):
    """Test that inference is the same as the original crepe"""
    activation = torchcrepe.infer(frames, 'full').detach().numpy()
    diff = np.abs(activation - activation_full)
    assert diff.max() < 1e-5 and diff.mean() < 1e-7
