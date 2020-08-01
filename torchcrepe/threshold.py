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
        pitch[harmonicity < self.value] = torchcrepe.UNVOICED
        return pitch
    
    
class Hysteresis:
    """Hysteresis thresholding"""
    
    def __init__(self, lower_bound=.19, upper_bound=.31, width=.065, stds=2.):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.width = width
        self.stds = stds
        
    def __call__(self, pitch, harmonicity):
        # Perform hysteresis in log-2 space
        pitch = torch.log2(pitch).detach().cpu().numpy()
        
        # Ignore confidently unvoiced pitch
        pitch[harmonicity < self.lower_bound] = dar.UNVOICED
        
        # Whiten pitch
        mean, std = np.nanmean, np.nanstd(pitch)
        pitch = (pitch - mean) / std
        
        # Require high confidence to make predictions far from the mean
        parabola = self.width * pitch ** 2 - self.width * self.stds ** 2
        threshold = lower_bound + np.clip(parabola, 0, 1 - self.lower_bound)
        threshold[np.isnan(threshold)] = lower_bound
        
        # Apply hysteresis to prevent short, unconfident voiced regions
        i = 0
        print(f'hysteresis length {len(hysteresis)}')
        while i < len(hysteresis) - 1:
            
            # Detect unvoiced to voiced transition
            if hysteresis[i] < threshold[i] and \
               hysteresis[i + 1] > threshold[i + 1]:
                
                # Grow region until next unvoiced or end of array
                start, end, keep = i + 1, i + 1, False
                while end < len(hysteresis) and \
                      hysteresis[end] > threshold[end]:
                    if hysteresis[end] > self.upper_bound:
                        keep = True
                    end += 1
                
                # Force unvoiced if we didn't pass the confidence required by the
                # hysteresis
                if not keep:
                    threshold[start:end] = 1
                
                i = end
                
            else:
                i += 1
        
        # Remove pitch with low harmonicity
        pitch[hysteresis < threshold] = torchcrepe.UNVOICED
        
        # Unwhiten
        pitch = pitch * std + mean
        
        # Convert to Hz
        return 2 ** pitch
