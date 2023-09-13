import torch

class MeanFilter(torch.nn.Module):
    def __init__(self, device, kernel_size=2):
        assert ((kernel_size - 1) % 2) == 0
        super().__init__()

        self.conv = torch.nn.Conv2d(1, 1, kernel_size=(kernel_size, kernel_size),
                padding=(kernel_size//2, kernel_size//2), bias=False).to(device)

        # Set the weights to represent a mean filter
        wd = torch.ones(1, 1, kernel_size, kernel_size).to(device)
        wd /= (kernel_size ** 2)
        self.conv.weight.data = wd
        self.eval()

    def forward(self, x):
        y = self.conv(x)
        return y

class MelLossAP():
    phase_filter: MeanFilter

    def __init__(self, device):
        self.phase_filter = MeanFilter(device, kernel_size=5)

    def norm_mels(self, y_mel_a, y_g_hat_mel_a):
        max_ampl_val = torch.max(y_mel_a, y_g_hat_mel_a).max()
        min_ampl_val = torch.min(y_mel_a, y_g_hat_mel_a).min()
        adiff = max_ampl_val - min_ampl_val
        y_mel_a = (y_mel_a - min_ampl_val) / adiff
        y_g_hat_mel_a = (y_g_hat_mel_a - min_ampl_val) / adiff
        return (y_mel_a, y_g_hat_mel_a)

    def get_loss(self, *args):
        for arg in args:
            assert self.phase_filter.conv.weight.data.device == arg.device
        y_mel_a, y_g_hat_mel_a, y_mel_p, y_g_hat_mel_p = args
        loss_phase = y_mel_p - y_g_hat_mel_p
        #print(1, loss_phase.size())
        loss_phase = torch.sin(loss_phase / 2)
        #print(2, loss_phase.size())
        loss_phase = self.phase_filter(loss_phase.unsqueeze(1)).squeeze(1)
        #print(3, loss_phase.size())
        loss_phase = loss_phase.abs()
        y_mel_a, y_g_hat_mel_a = self.norm_mels(y_mel_a,
            y_g_hat_mel_a)
        max_ampl = torch.max(y_mel_a, y_g_hat_mel_a)
        loss_phase = loss_phase * max_ampl
        assert loss_phase.min() >= 0.0 and loss_phase.max() <= 1.0
        loss_mel = torch.abs(y_mel_a - y_g_hat_mel_a) + loss_phase
        loss_freq = loss_mel.sum(dim=1).mean()
        loss_time = loss_mel.sum(dim=2).mean()
        loss_mel = (0.5 * loss_freq + 0.5 * loss_time)
        return (loss_mel, loss_freq, loss_time)
