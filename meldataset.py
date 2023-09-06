import hashlib
import math
import os
import random
import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from scipy.signal import firwin
from librosa.filters import mel as librosa_mel_fn
from transformers import SpeechT5Processor as ST5P, \
        SpeechT5ForSpeechToSpeech as ST5FSTS, \
        SpeechT5HifiGan as ST5HG

MAX_WAV_VALUE = 32768.0


def get_PBF(fs=16000, l_cut=75.0, h_cut=4000.0):
    numtaps = 1023  # Number of filter taps (coefficients)
    coeffs = firwin(numtaps, [l_cut, h_cut], pass_zero='bandpass', fs=fs)

    # Convert to PyTorch tensor
    filter_kernel = torch.tensor(coeffs, dtype=torch.float32).view(1, 1, -1)
    return filter_kernel


def load_wav(full_path, do_normalize):
    sampling_rate, data = read(full_path)
    data = data / MAX_WAV_VALUE
    if do_normalize:
        data = normalize(data) * 0.95

    audio = torch.FloatTensor(data)
    audio = audio.unsqueeze(0)
    o_flt = get_PBF(fs=sampling_rate)
    audio_flt = F.conv1d(audio.unsqueeze(0), o_flt, padding=(o_flt.size(2) - 1) // 2)
    return audio, audio_flt.squeeze(0), sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

nclamps_MAX = 10
_nclamps = 0


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax,
                    center=False, return_phase=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))
        global _nclamps
        assert y.max() < 1.1 and _nclamps < nclamps_MAX
        _nclamps += 1
        y = y.clamp(max=1.)

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = F.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    if not return_phase:
        specs = [torch.sqrt(spec.pow(2).sum(-1)+(1e-9)),]
    else:
        amplitude = torch.sqrt(spec[..., 0]**2 + spec[..., 1]**2)
        phase = torch.atan2(spec[..., 1], spec[..., 0])
        # Normalize
        phase = (phase + torch.tensor(math.pi)) / (2 * torch.tensor(math.pi))
        specs = [amplitude, phase]
        #assert amplitude.size() == phase.size()
        #print(f'amplitude.size() = {amplitude.size()}')
        #print(f'phase.size() = {phase.size()}')

    specs[0] = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], specs[0])
    specs[0] = spectral_normalize_torch(specs[0])

    if not return_phase:
        specs = specs[0]

    return specs


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files[:600], validation_files[:64]


def hash_filename(filename):
    # Create a new sha256 hash object
    sha256 = hashlib.sha256()

    # Update the hash object with the bytes of the filename
    sha256.update(filename.encode())

    # Get the hexadecimal representation of the digest
    hashed_filename = sha256.hexdigest()

    return hashed_filename


class MelDataset(torch.utils.data.Dataset):
    cache_dir = os.path.expanduser('~/.cache/hifi-gan')
    processor = None
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = {}
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path


    def getMelRef(self, audio_in, filename):
        fn_hash = hash_filename(f'v2-{filename}')
        o_fn_hash = hash_filename(f'v1-{filename}')
        for i, _fn_hash in enumerate((fn_hash, o_fn_hash)):
            mels_cp = os.path.join(self.cache_dir, f"{_fn_hash}.mel.pt")
            audios_cp = os.path.join(self.cache_dir, f"{_fn_hash}.audio.pt")
            if os.path.exists(mels_cp) and os.path.exists(audios_cp):
                if i == 1 and np.random.rand() < 1.05:
                    continue
                mels = torch.load(mels_cp, map_location = 'cpu')
                audios = torch.load(audios_cp, map_location = 'cpu')
                if i == 0:
                    for xx in ("mel", "audio"):
                        ofn = os.path.join(self.cache_dir, f"{o_fn_hash}.{xx}.pt")
                        try:
                            os.unlink(ofn)
                        except FileNotFoundError:
                            pass
                return (audios, mels)
        mels_cp = os.path.join(self.cache_dir, f"{fn_hash}.mel.pt")
        audios_cp = os.path.join(self.cache_dir, f"{fn_hash}.audio.pt")
        print(f'getMelRef({filename}, {audio_in.size()}')

        if self.processor is None:
            self.processor = ST5P.from_pretrained("microsoft/speecht5_vc")
            model = ST5FSTS.from_pretrained("microsoft/speecht5_vc")
            model.eval()
            self.model = model.to(self.device)
            vocoder = ST5HG.from_pretrained("microsoft/speecht5_hifigan")
            vocoder.eval()
            self.vocoder = vocoder.to(self.device)
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)

        speaker_embeddings = torch.randn(1, 512, device = self.model.device)
        try:
            inputs = self.processor(audio=audio_in.squeeze(0), sampling_rate=self.sampling_rate,
                                    return_tensors="pt")
            inputs = inputs["input_values"].to(self.model.device)
            print(f'inputs.size() = {inputs.size()}')
            mel = self.model.generate_speech(inputs, speaker_embeddings)
        except RuntimeError:
            print(f'getMelRef({audio_in.size()}')
            raise
        audio = self.vocoder(mel).unsqueeze(0).cpu()
        mel = mel.t().unsqueeze(0).cpu()
        torch.save(audio, audios_cp)
        torch.save(mel, mels_cp)
        return (audio, mel)


    @torch.no_grad()
    def __getitem__(self, index):
        #assert not self.split
        filename = self.audio_files[index]
        if not filename in self.cached_wav:
            #print(f'cache miss {filename}, cache_len={len(self.cached_wav.keys())}')
            audio, audio_flt, sampling_rate = load_wav(filename, not self.fine_tuning)
            if self.fine_tuning:
                audio, mel = self.getMelRef(audio, filename)
            self.cached_wav[filename] = (audio, mel, filename)
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio, mel, _filename = self.cached_wav[filename]
            assert _filename == filename
            #self._cache_ref_count -= 1

        if not self.fine_tuning:
            assert (audio.size() == audio_flt.size())

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                    #audio_flt = audio_flt[:, audio_start:audio_start + \
                    #                      self.segment_size]
                else:
                    pad_size = (0, self.segment_size - audio.size(1))
                    audio = F.pad(audio, pad_size, 'constant')
                    #audio_flt = F.pad(audio_flt, pad_size, 'constant')

            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
        else:
            #mel = np.load(
            #    os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            #mel = torch.from_numpy(mel)

            #if len(mel.shape) < 3:
            #    mel = mel.unsqueeze(0)
            # mel = audio_flt
            #audio_flt = audio

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel_start_x = random.randint(0, int((mel.size(2) - frames_per_seg - 1) / 2)) * 2
                    assert mel_start_x % 2 == 0 and mel_start_x + frames_per_seg < mel.size(2)
                    #print(f'mel_start={mel_start_x}, mel.size(2)={mel.size(2)}')
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = F.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = F.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
                #mel = mel.to(self.device)
                #audio = audio.to(self.device)

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels*4,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False, return_phase=True)
        #print(mel.size())#, mel.device, mel_loss.size(), mel_loss.device)
        #raise Exception("BP")
        return (mel.squeeze(), audio.squeeze(0), filename,
                (mel_loss[0].squeeze(), mel_loss[1].squeeze()))


    def __len__(self):
        return len(self.audio_files)
