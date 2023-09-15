from scipy.signal import firwin
import torch


def get_PBF(fs=16000, l_cut=75.0, h_cut=4000.0):
    numtaps = 1023  # Number of filter taps (coefficients)
    coeffs = firwin(numtaps, [l_cut, h_cut], pass_zero='bandpass', fs=fs)

    # Convert to PyTorch tensor
    filter_kernel = torch.tensor(coeffs, dtype=torch.float32).view(1, 1, -1)
    return filter_kernel
