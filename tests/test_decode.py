import torch
import torchcrepe
import pytest


###############################################################################
# Test decode.py
###############################################################################


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA device")
def test_weighted_argmax_decode():
    """Tests that weighted argmax decode works without CUDA assertion error"""
    fake_logits = torch.rand(8, 360, 128, device="cuda")
    decoded = torchcrepe.decode.weighted_argmax(fake_logits)


def test_viterbi_banded_fast_matches_legacy():
    generator = torch.Generator().manual_seed(0)
    for frames in (16, 128, 1024):
        probabilities = torch.rand((2, torchcrepe.PITCH_BINS, frames),
                                   generator=generator)
        legacy_bins, legacy_pitch = torchcrepe.decode.viterbi_legacy(
            probabilities, dither=False)
        fast_bins, fast_pitch = torchcrepe.decode.viterbi_banded_fast(
            probabilities, dither=False)
        assert torch.equal(legacy_bins, fast_bins)
        assert torch.allclose(legacy_pitch, fast_pitch)


def test_viterbi_alias_preserves_legacy_path():
    generator = torch.Generator().manual_seed(1)
    probabilities = torch.rand((1, torchcrepe.PITCH_BINS, 256),
                               generator=generator)
    legacy_bins, legacy_pitch = torchcrepe.decode.viterbi_legacy(
        probabilities, dither=False)
    bins, pitch = torchcrepe.decode.viterbi(probabilities, dither=False)
    assert torch.equal(legacy_bins, bins)
    assert torch.allclose(legacy_pitch, pitch)
