import numpy as np
import torch

import torchcrepe


###############################################################################
# Pitch thresholding methods
###############################################################################


class At:
    """Simple thresholding at a specified probability value"""

    def __init__(self, value):
        self.value = value

    def __call__(self, pitch, harmonicity):
        # Make a copy to prevent in-place modification
        pitch = torch.clone(pitch)

        # Threshold
        pitch[harmonicity < self.value] = torchcrepe.UNVOICED
        return pitch


class Hysteresis:
    """Hysteresis thresholding"""

    def __init__(self,
                 lower_bound=.19,
                 upper_bound=.31,
                 width=.2,
                 stds=1.7,
                 return_threshold=False):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.width = width
        self.stds = stds
        self.return_threshold = return_threshold

    def __call__(self, pitch, harmonicity):
        # Save output device
        device = pitch.device

        # Perform hysteresis in log-2 space
        pitch = torch.log2(pitch).detach().flatten().cpu().numpy()

        # Flatten harmonicity
        harmonicity = harmonicity.flatten().cpu().numpy()

        # Ignore confidently unvoiced pitch
        pitch[harmonicity < self.lower_bound] = torchcrepe.UNVOICED

        # Whiten pitch
        mean, std = np.nanmean(pitch), np.nanstd(pitch)
        pitch = (pitch - mean) / std

        # Require high confidence to make predictions far from the mean
        parabola = self.width * pitch ** 2 - self.width * self.stds ** 2
        threshold = \
            self.lower_bound + np.clip(parabola, 0, 1 - self.lower_bound)
        threshold[np.isnan(threshold)] = self.lower_bound

        # Apply hysteresis to prevent short, unconfident voiced regions
        i = 0
        while i < len(harmonicity) - 1:

            # Detect unvoiced to voiced transition
            if harmonicity[i] < threshold[i] and \
               harmonicity[i + 1] > threshold[i + 1]:

                # Grow region until next unvoiced or end of array
                start, end, keep = i + 1, i + 1, False
                while end < len(harmonicity) and \
                      harmonicity[end] > threshold[end]:
                    if harmonicity[end] > self.upper_bound:
                        keep = True
                    end += 1

                # Force unvoiced if we didn't pass the confidence required by
                # the hysteresis
                if not keep:
                    threshold[start:end] = 1

                i = end

            else:
                i += 1

        # Remove pitch with low harmonicity
        pitch[harmonicity < threshold] = torchcrepe.UNVOICED

        # Unwhiten
        pitch = pitch * std + mean

        # Convert to Hz
        pitch = torch.tensor(2 ** pitch, device=device)[None, :]

        # Optionally return threshold
        if self.return_threshold:
            return pitch, torch.tensor(threshold, device=device)

        return pitch
