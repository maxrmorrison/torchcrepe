import torch

import torchcrepe


###############################################################################
# Test threshold.py
###############################################################################


def test_at():
    """Test torchcrepe.threshold.At"""
    input_pitch = torch.tensor([100., 110., 120., 130., 140.])
    periodicity = torch.tensor([.19, .22, .25, .17, .30])

    # Perform thresholding
    output_pitch = torchcrepe.threshold.At(.20)(input_pitch, periodicity)

    # Ensure thresholding is not in-place
    assert not (input_pitch == output_pitch).all()

    # Ensure certain frames are marked as unvoiced
    isnan = torch.isnan(output_pitch)
    assert isnan[0] and isnan[3]
    assert not isnan[1] and not isnan[2] and not isnan[4]
