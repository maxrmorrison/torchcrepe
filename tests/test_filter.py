import timeit

import torch

from torchcrepe.filter import mean, median, nanfilter, nanmean, nanmedian

###############################################################################
# Test filter.py
###############################################################################


def test_mean():
    _deprecated_mean = lambda x, win_length: nanfilter(x, win_length, nanmean)

    x = torch.rand(1, 44100)
    x[torch.rand_like(x) < 0.1] = float("nan")

    assert torch.allclose(mean(x, 3), _deprecated_mean(x, 3), equal_nan=True)
    assert torch.allclose(mean(x, 9), _deprecated_mean(x, 9), equal_nan=True)

    # time_mean = timeit.timeit(lambda: mean(x, 3), number=10)
    # time_deprecated_mean = timeit.timeit(lambda: _deprecated_mean(x, 3), number=10)

    # print(
    #     f"mean: {time_mean}, deprecated_mean: {time_deprecated_mean}, speed: {time_deprecated_mean / time_mean}x"
    # )


def test_median():
    _deprecated_median = lambda x, win_length: nanfilter(x, win_length, nanmedian)

    x = torch.rand(1, 44100)
    x[torch.rand_like(x) < 0.1] = float("nan")

    assert torch.allclose(median(x, 3), _deprecated_median(x, 3), equal_nan=True)
    assert torch.allclose(median(x, 9), _deprecated_median(x, 9), equal_nan=True)

    # time_median = timeit.timeit(lambda: median(x, 3), number=10)
    # time_deprecated_median = timeit.timeit(lambda: _deprecated_median(x, 3), number=10)

    # print(
    #     f"median: {time_median}, deprecated_median: {time_deprecated_median}, speed: {time_deprecated_median / time_median}x"
    # )
