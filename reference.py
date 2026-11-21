import numpy as np


def impl(x, kernel_size, stride):
    ks = int(kernel_size)
    st = int(stride)
    windows = np.lib.stride_tricks.sliding_window_view(x, (
        ks,
        ks
    ), axis=(1, 2))
    windows = windows[:, ::st, ::st]
    y = windows.mean(axis=(-2, -1))
    return y.astype(x.dtype)
