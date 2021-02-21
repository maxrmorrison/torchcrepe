import pytest
import torch
import torchcrepe


###############################################################################
# Test decode.py
###############################################################################


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA device")
def test_weighted_argmax_decode():
    """Tests that weighted argmax decode works without CUDA assertion error"""
    fake_logits = torch.rand(8, 360, 128, device="cuda")
    decoded = torchcrepe.decode.weighted_argmax(fake_logits)
