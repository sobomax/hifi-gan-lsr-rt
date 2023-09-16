import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt
from numpy import abs as np_abs


def plot_spectrogram(spectrogram, *im_a, **im_kw):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none',
                   *im_a, **im_kw)
    cbar = plt.colorbar(im, ax=ax)
    cbar_min = spectrogram.min()
    cbar_max = spectrogram.max()
    cbar_mean = spectrogram.mean()
    format_spec = ".2f" if np_abs(cbar_max - cbar_min) > 0.1 else ".2e"
    lb = f"min: {cbar_min:{format_spec}}\n" \
            f"max: {cbar_max:{format_spec}}\n" \
            f"mean: {cbar_mean:{format_spec}}"
    cbar.set_label(lb)
    cbar.ax.yaxis.label.set_rotation(0)
    cbar.ax.yaxis.label.set_position((40, 1))
    cbar.ax.yaxis.label.set_horizontalalignment('left')

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def anomaly_check(x, name):
    if x.isnan().any():
        raise Exception(f"{name} anomaly: NaN: {x}")
    if x.isinf().any():
        raise Exception(f"{name} anomaly: Infinity detected: {x}")
