
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
try:
    import intel_extension_for_pytorch as ipex
except ModuleNotFoundError:
    ipex = None
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, mel_spec_options, \
        get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
from mel_loss import MelLossAP
from matplotlib.colors import LogNorm

import multiprocessing
from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig
#torch.backends.cudnn.benchmark = True

debug_m = False

class MySpeechT5HifiGan(SpeechT5HifiGan):
    _frame_size = 256 # Fixme
    sampling_rate = None
    pre_frames = 2
    post_frames = 2
    def __init__(self, h=None):
        self.sampling_rate = h.sampling_rate
        st5conf = SpeechT5HifiGanConfig(
                #model_in_dim=81,
                )
        return super().__init__(st5conf)


    def forward(self, x, debug=False, chunks=None):
        if debug:
            print(f'MySpeechT5HifiGan.forward(x.size = {x.size()})')
        x = x.permute(0, 2, 1)
        prfs = torch.zeros(x.size(0), self.pre_frames, x.size(2),
                           device=x.device)
        pofs = torch.zeros(x.size(0), self.post_frames, x.size(2),
                           device=x.device)
        x = torch.cat((prfs, x, pofs), dim=1)
        y_trim_pr = self.pre_frames * self._frame_size
        y_trim_po = self.post_frames * self._frame_size
        if not self.training and chunks is None:
            y = super().forward(x)
            return y[:, y_trim_pr:-y_trim_po]
        if debug:
            print(f'x.size = {x.size()}')
        if chunks is None:
            chunks = (4,)
        z = []
        for chunk_size in chunks:
            y = []
            _x = x.clone()
            if debug:
                print(chunk_size)
            eframes = self.pre_frames + self.post_frames
            assert _x.size(1) > eframes
            extra_pad = (_x.size(1) - eframes) % chunk_size
            assert extra_pad < chunk_size
            if extra_pad > 0:
                _pofs = torch.zeros(x.size(0), extra_pad,
                                    x.size(2), device=x.device)
                _x = torch.cat((_x, _pofs), dim=1)
            assert ((_x.size(1) - eframes) % chunk_size) == 0
            while _x.size(1) > eframes:
                chunk = _x[:, :chunk_size+eframes, :]
                assert chunk.size(1) == chunk_size+eframes
                _y = super().forward(chunk)
                y.append(_y[:, y_trim_pr:-y_trim_po])
                _x = _x[:, chunk_size:, :]
            if extra_pad > 0:
                ep_trim = extra_pad * self._frame_size
                assert y[-1].size(1) > ep_trim
                y[-1] = y[-1][:, :-ep_trim]
            z.append(torch.cat(y, dim=1))

        return z


class MyMSO(mel_spec_options):
    def __init__(self, h, a):
        super().__init__()
        self.n_fft = h.n_fft
        self.num_mels = h.num_mels * a.mel_oversample
        self.sampling_rate = h.sampling_rate
        self.hop_size = h.hop_size
        self.win_size = h.win_size
        self.fmin = h.fmin
        self.fmax = h.fmax


class TrainingMSO(MyMSO):
    def __init__(self, h, a):
        super().__init__(h, a)
        self.fmax = h.fmax_for_loss
        self.return_phase = True

class MyTrainer():
    def train(self, rank, a, h):
        if h.num_gpus > 1:
            init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                            world_size=h.dist_config['world_size'] * h.num_gpus, rank=self.rank)

        torch.manual_seed(h.seed)
        self.h = h
        self.a = a
        self.rank = rank
        if self.rank == 0 and a.device != 'cpu':
            if not ':' in a.device:
                devname = f'{a.device}:{self.rank}'
            else:
                devname = a.device
        else:
            devname = 'cpu'
        self.device = device = torch.device(devname)

        self.generator = MySpeechT5HifiGan(h).to(device)
        if not a.generator_only:
            self.mpd = MultiPeriodDiscriminator().to(device)
            self.msd = MultiScaleDiscriminator().to(device)
        else:
            self.mpd = None
            self.msd = None

        if self.rank == 0:
            print(self.generator)
            os.makedirs(a.checkpoint_path, exist_ok=True)
            print("checkpoints directory : ", a.checkpoint_path)

        if os.path.isdir(a.checkpoint_path):
            cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
            cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

        self.steps = 0
        last_epoch = -1
        self.min_yghe_value = None
        self.max_yghe_value = None
        if cp_g is None:
            state_dict_g = None
            try:
                self.generator = MySpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
            except RuntimeError:
                pass
        else:
            state_dict_g = load_checkpoint(cp_g, device)
            self.generator.load_state_dict(state_dict_g['generator'])
            del state_dict_g['generator']
            self.steps = state_dict_g['steps'] + 1
            last_epoch = state_dict_g['epoch']
            self.min_yghe_value = state_dict_g['min_yghe_value']
            self.max_yghe_value = state_dict_g['max_yghe_value']
        if cp_do is None:
            state_dict_do = None
        else:
            state_dict_do = load_checkpoint(cp_do, device)
            if self.mpd:
                self.mpd.load_state_dict(state_dict_do['mpd'])
            if self.msd:
                self.msd.load_state_dict(state_dict_do['msd'])
            self.steps = state_dict_do['steps'] + 1
            last_epoch = state_dict_do['epoch']

        if h.num_gpus > 1:
            self.generator = DistributedDataParallel(self.generator,
                    device_ids=[self.rank]).to(device)
            if self.mpd:
                self.mpd = DistributedDataParallel(self.mpd, device_ids=[self.rank]).to(device)
            if self.msd:
                self.msd = DistributedDataParallel(self.msd, device_ids=[self.rank]).to(device)

        self.optim_g = torch.optim.AdamW(self.generator.parameters(), h.learning_rate,
                betas=[h.adam_b1, h.adam_b2])
        if self.mpd:
            assert self.msd is not None
            self.optim_d = torch.optim.AdamW(itertools.chain(self.msd.parameters(), self.mpd.parameters()),
                                    h.learning_rate, betas=[h.adam_b1, h.adam_b2])

        if state_dict_do is not None:
            self.optim_d.load_state_dict(state_dict_do['optim_d'])
            self.optim_g.load_state_dict(state_dict_do['optim_g'])
        elif state_dict_g is not None:
            self.optim_g.load_state_dict(state_dict_g['optim_g'])
            del state_dict_g['optim_g']

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
        if self.mpd:
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

        training_filelist, validation_filelist = get_dataset_filelist(a)

        mso_ref = MyMSO(h, a)
        self.mso_loss = TrainingMSO(h, a)
        trainset = MelDataset(training_filelist, h.segment_size, h.hop_size, h.sampling_rate,
                            mso_ref, self.mso_loss, n_cache_reuse=1024,
                            shuffle=False if h.num_gpus > 1 else True, device=device,
                            fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

        train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

        self.train_loader = DataLoader(trainset, num_workers=h.num_workers,
                                shuffle=a.shuffle_input,
                                prefetch_factor=3,
                                sampler=train_sampler,
                                batch_size=h.batch_size,
                                pin_memory=a.pin_memory,
                                drop_last=True if not a.precompute_mels else False,
                                persistent_workers=True if h.num_workers > 0 else False)

        if a.precompute_mels:
            for i, batch in enumerate(self.train_loader):
                print(f'{i}: {len(batch) * len(batch[0])}')
            exit(0)
        self.mel_loss_o = MelLossAP(device)

        if self.rank == 0:
            validset = MelDataset(validation_filelist, h.segment_size, h.hop_size, h.sampling_rate,
                                    mso_ref, self.mso_loss, split=False, shuffle=False, n_cache_reuse=1024,
                                    device=device, fine_tuning=a.fine_tuning,
                                    base_mels_path=a.input_mels_dir)
            nw = min(1, h.num_workers)
            self.validation_loader = DataLoader(validset, num_workers=nw, shuffle=False,
                                        prefetch_factor=3,
                                        sampler=None,
                                        batch_size=1,
                                        pin_memory=a.pin_memory,
                                        drop_last=True,
                                        persistent_workers=True if nw > 0 else False)

            sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

        #torch.autograd.set_detect_anomaly(True)
        torch.cuda.empty_cache()
        self.generator.train()
        if self.mpd:
            self.mpd.train()
        if self.msd:
            self.msd.train()

        if ipex is not None:
            self.generator, self.optim_g = ipex.optimize(model=self.generator,
                    optimizer=self.optim_g)
        for epoch in range(max(0, last_epoch), a.training_epochs):
            if self.rank == 0:
                start = time.time()
                print("epoch: {}".format(self.epoch+1))

            if self.steps == 0 and a.fine_tuning:
                self.validate(sw)

            if h.num_gpus > 1:
                train_sampler.set_epoch(self.epoch)

            for i, batch in enumerate(self.train_loader):
                loss_gen_all, loss_mel, loss_mel_a, loss_mel_p = self.train_step(i, batch, sw)

                if self.rank != 0:
                    self.steps += 1
                    continue

                # STDOUT logging
                stdur = time.time() - self.start_b
                with torch.no_grad():
                    mel_error = loss_mel.item()
                if self.steps % a.stdout_interval == 0:
                    #with torch.no_grad():
                    #    mel_error = loss_mel.item()
                    #    #mel_error += F.l1_loss(y_mel[1], y_g_hat_mel[1]).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, ' \
                            'Mel-Spec. Error : {:4.3f} ({:4.3f}A+{:4.3f}P), ' \
                            's/b : {:4.3f}'.
                        format(self.steps, loss_gen_all, mel_error, loss_mel_a,
                                loss_mel_p, stdur))

                # checkpointing
                if self.steps % a.checkpoint_interval == 0 and self.steps != 0:
                    self.save_checkpoint(sw)

                # Tensorboard summary logging
                if self.steps % a.summary_interval == 0:
                    self.do_summary(sw, loss_gen_all, mel_error, stdur)

                # Validation
                if self.steps % a.validation_interval == 0 and self.steps != 0:
                    self.validate(sw)

                self.steps += 1

            scheduler_g.step()
            if self.mpd:
                scheduler_d.step()

            if self.rank == 0:
                epdur = time.time() - start
                sw.add_scalar("performance/secs_per_epoch", epdur, self.epoch)
                print('Time taken for epoch {} is {} sec\n'.format(self.epoch + 1, int(epdur)))
                sw.flush()

    def train_step(self, i, batch, sw):
        if self.rank == 0:
            self.start_b = time.time()
        x, y, fn, y_mel = batch
        x = x.to(self.device)
        y = y.to(self.device)
        y_mel = [ym.to(self.device) for ym in y_mel]

        y = y.unsqueeze(1)

        y_g_hats = self.generator(x)
        y_g_hat_mels = [mel_spectrogram(ygh.squeeze(1), self.mso_loss)
                        for ygh in y_g_hats]

        # MPD
        if self.mpd:
            self.optim_d.zero_grad()
            y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        if self.msd:
            y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        if self.msd:
            loss_disc_all = loss_disc_s
        if self.mpd:
            loss_disc_all += loss_disc_f

        if self.mpd:
            loss_disc_all.backward()
            self.optim_d.step()

        # Generator
        self.optim_g.zero_grad()

        # Compute Mel-Spectrogram Loss
        y_mel_a = y_mel[0]
        y_mel_a = y_mel_a.repeat(len(y_g_hat_mels), 1, 1)
        y_g_hat_mel_a = torch.cat([yghm.mel_a for yghm in y_g_hat_mels],
                dim=0)
        y_mel_p = y_mel[1]
        y_mel_p = y_mel_p.repeat(len(y_g_hat_mels), 1, 1)
        y_g_hat_mel_p = torch.cat([yghm.mel_p for yghm in y_g_hat_mels],
                dim=0)
        loss_mel = self.mel_loss_o.get_loss(y_mel_a, y_g_hat_mel_a,
                y_mel_p, y_g_hat_mel_p)
        loss_mel, loss_mel_a, loss_mel_p = loss_mel

        if debug_m:
            print(loss_mel)

        if y_g_hat_mels[0].mel_a.isnan().any():
            raise Exception(f"NaN: y_g_hat_mels[0].mel_a: {y_g_hat_mels[0].mel_a}")
        if loss_mel.isnan().any():
            raise Exception(f"NaN: loss_mel: {loss_mel}, {loss_mel_a}, {loss_mel_p}")
        #print(f'y_g_hat_mel[0].size() = {y_g_hat_mel[0].size()}')

        if self.mpd:
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
        if self.msd:
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
        if self.mpd:
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        if self.msd:
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        if self.mpd:
            loss_gen_f, _ = generator_loss(y_df_hat_g)
        if self.msd:
            loss_gen_s, _ = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_mel
        if self.mpd:
            loss_gen_all += loss_fm_f + loss_gen_f
        if self.msd:
            loss_gen_all += loss_fm_s + loss_gen_s

        loss_gen_all.backward()
        self.optim_g.step()
        if self.steps == 1695 and False:
            sw.add_audio(f'outliers/y_{fn}', y[0], self.steps, self.h.sampling_rate),
            sw.add_audio(f'outliers/y_g_hats_{fn}', y_g_hats[0], self.steps, self.h.sampling_rate),
            print(fn, loss_mel)
            sw.flush()
        return loss_gen_all, loss_mel, loss_mel_a, loss_mel_p


    def validate(self, sw):
        self.generator.eval()
        torch.cuda.empty_cache()
        val_err_tot = 0
        with torch.no_grad():
            yghes = []
            for j, batch in enumerate(self.validation_loader):
                x, y, fn, y_mel = batch
                fn = os.path.basename(fn[0])
                chunksz = 4 #min(2 ** (j + 2), 32)
                y_g_hat = self.generator(x.to(self.device), chunks=(chunksz,))[0]
                y_mel = [ym.to(self.device) for ym in y_mel]
                y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), self.mso_loss)
                vet_inc = F.l1_loss(y_mel[0], y_g_hat_mel.mel_a,
                                    reduction='sum') / y_mel[0].numel()
                vet_inc = vet_inc.item()
                val_err_tot += vet_inc

                if j <= 4:
                    yghe_name = f'generated/y_hat_err_{j}'
                    yghe_data = (y_mel[0] - y_g_hat_mel.mel_a).abs()
                    if debug_m:
                        for k, z in enumerate((y_mel[1], y_g_hat_mel.mel_p)):
                            print(f'k:', z.min(), z.max(), z.mean())
                    yghe_data_ph = y_mel[1] - y_g_hat_mel.mel_p
                    #print(yghe_data_ph.min(), yghe_data_ph.max(), yghe_data_ph.mean())
                    # yghe_data_ph / 2 here is so pi difference prodice max
                    # loss, which sin has at pi/2
                    yghe_data_ph = torch.sin(yghe_data_ph / 2)
                    yghe_data_ph = yghe_data_ph.unsqueeze(0)
                    yghe_data_ph = self.mel_loss_o.phase_filter(yghe_data_ph).squeeze().abs()
                    #print(y_mel[0].size(), y_g_hat_mel.mel_a.size())
                    y_mel_an, y_g_hat_mel_an = self.mel_loss_o.norm_mels(
                            y_mel[0], y_g_hat_mel.mel_a)
                    max_ampl = torch.max(y_mel_an, y_mel_an).squeeze()
                    yghe_data_ph *= max_ampl
                    #print(yghe_data_ph.size())
                    #print(yghe_data_ph.min(), yghe_data_ph.max(), yghe_data_ph.mean())
                    if debug_m:
                        print(yghe_data_ph.size())
                    yghe_data = yghe_data.squeeze(0)
                    yghes.append((yghe_name, yghe_data, yghe_data_ph))

                    sw.add_text(f'input/filename_{j}', fn, self.steps)
                    sw.add_audio(f'gt/y_{j}', y[0], self.steps, self.h.sampling_rate),
                    sw.add_figure(f'gt/y_spec_{j}', plot_spectrogram(y_mel[0][0].cpu().numpy()),
                                  self.steps)

                    sw.add_audio(f'generated/y_hat_{j}', y_g_hat[0], self.steps,
                                 self.h.sampling_rate)
                    gx_name = f'input/generator_x_{j}'
                    gx_data = x.squeeze(0).cpu().numpy()
                    gx_spec = plot_spectrogram(gx_data)
                    sw.add_figure(gx_name, gx_spec, self.steps)
                    mso = MyMSO(self.h, self.a)
                    mso.return_phase = False
                    y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), mso).mel_a
                    yhs_name = f'generated/y_hat_spec_{j}'
                    yhs_data = y_hat_spec.squeeze(0).cpu().numpy()
                    yhs_spec = plot_spectrogram(yhs_data)
                    sw.add_figure(yhs_name, yhs_spec, self.steps)
            if self.min_yghe_value is None:
                self.min_yghe_value = min([data[1].min().cpu().numpy()
                                     for data in yghes])
            if self.max_yghe_value is None:
                self.max_yghe_value = max([data[1].max().cpu().numpy()
                                     for data in yghes])
            for j, yghe in enumerate(yghes):
                yghe_name, yghe_data, yghe_data_ph = yghe
                yghe_norm_min = max(self.min_yghe_value, self.max_yghe_value / 100)
                yghe_norm = LogNorm(vmin=yghe_norm_min,
                                    vmax=self.max_yghe_value)
                yghe_spec = plot_spectrogram(yghe_data.cpu().numpy(),
                                             cmap='plasma',
                                             norm=yghe_norm)
                sw.add_figure(yghe_name, yghe_spec, self.steps)
                yghe_spec_ph = plot_spectrogram(yghe_data_ph.cpu().numpy(),
                                    cmap='inferno')
                sw.add_figure(f'generated/y_hat_err_ph_{j}',
                              yghe_spec_ph, self.steps)

            val_err = val_err_tot / (j+1)
            sw.add_scalar("validation/mel_spec_error", val_err, self.steps)
            print(f'Validation error: {val_err:4.3f}')
        sw.flush()
        self.generator.train()

    def save_checkpoint(self, sw):
        checkpoint_path = "{}/g_{:08d}".format(self.a.checkpoint_path, self.steps)
        cp_data = {'generator': (self.generator.module
                                if self.h.num_gpus > 1
                                else self.generator).state_dict(),
                'min_yghe_value': self.min_yghe_value,
                'max_yghe_value': self.max_yghe_value}
        if self.mpd is None:
            cp_data['optim_g'] = self.optim_g.state_dict()
            cp_data['steps'] = self.steps
            cp_data['epoch'] = self.epoch
        save_checkpoint(checkpoint_path, cp_data)
        if self.mpd is not None or self.msd is not None:
            checkpoint_path = "{}/do_{:08d}".format(self.a.checkpoint_path, self.steps)
            cp_data = {}
            if self.mpd:
                mpd_s = (self.mpd.module if self.h.num_gpus > 1
                        else self.mpd).state_dict()
                cp_data['mpd'] = mpd_s
                cp_data['optim_d'] = self.optim_d.state_dict()
            if self.msd:
                cp_data['msd'] = (self.msd.module if self.h.num_gpus > 1
                        else self.msd).state_dict()
            cp_data['optim_g'] = self.optim_g.state_dict()
            cp_data['steps'] = self.steps
            cp_data['epoch'] = self.epoch

            save_checkpoint(checkpoint_path, cp_data)

    def do_summary(self, sw, loss_gen_all, mel_error, stdur):
        sw.add_scalar("training/gen_loss_total", loss_gen_all, self.steps)
        sw.add_scalar("training/mel_spec_error", mel_error, self.steps)
        sw.add_scalar("performance/secs_per_batch", stdur, self.steps)
        for i, param_group in enumerate(self.optim_g.param_groups):
            sw.add_scalar(f"training/optim_g_lr_{i}",
                          param_group['lr'], self.steps)
        sw.flush()


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=500, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1500, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--generator_only', default=False, type=bool)
    parser.add_argument('--precompute_mels', default=False, type=bool)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pin_memory', default=True, type=bool)
    parser.add_argument('--mel_oversample', default=1, type=int)
    parser.add_argument('--shuffle_input', default=True, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        #h.num_gpus = 2
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    trainer = MyTrainer()
    if h.num_gpus > 1:
        mp.spawn(trainer.train, nprocs=h.num_gpus, args=(a, h,))
    else:
        #multiprocessing.set_start_method('spawn', force=True)
        trainer.train(0, a, h)


if __name__ == '__main__':
    main()
