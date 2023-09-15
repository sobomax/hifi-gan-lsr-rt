from scipy.signal import firwin
import torch

class PassBandFilter(torch.nn.Module):
    numtaps = 1023

    def __init__(self, fs=16000, l_cut=75.0, h_cut=4000.0):
        super().__init__()
        coeffs = firwin(self.numtaps, [l_cut, h_cut], pass_zero='bandpass', fs=fs)
        padding = (len(coeffs) -1) // 2
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=1,
                kernel_size=len(coeffs),
                padding=padding, bias=False)
        # Load the filter coefficients into the Conv1d layer
        filter_kernel = torch.tensor(coeffs, dtype=torch.float32).view(1, 1, -1)
        self.conv1.weight.data = filter_kernel

    def forward(self, x):
        return self.conv1(x)

def get_PBF(fs=16000, l_cut=75.0, h_cut=4000.0):
    numtaps = 1023  # Number of filter taps (coefficients)
    coeffs = firwin(numtaps, [l_cut, h_cut], pass_zero='bandpass', fs=fs)

    # Convert to PyTorch tensor
    filter_kernel = torch.tensor(coeffs, dtype=torch.float32).view(1, 1, -1)
    return filter_kernel
