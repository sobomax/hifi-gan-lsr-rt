
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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

import multiprocessing
from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig
#torch.backends.cudnn.benchmark = True


class MySpeechT5HifiGan(SpeechT5HifiGan):
    def __init__(self, h):
        st5conf = SpeechT5HifiGanConfig()
        return super().__init__(st5conf)
    #def __new__(cls, *args, **kwargs):
    #    instance = super().from_pretrained("microsoft/speecht5_hifigan")
    #    return instance

    def forward(self, x, debug=False, chunks=None):
        x = x.permute(0, 2, 1)
        if not self.training and chunks is None:
            return super().forward(x)
        if debug:
            print(f'x.size = {x.size()}')
        if chunks is None:
            chunks = (32, 2, 4, 8, 16)
        z = []
        for chunk_size in chunks:
            y = []
            _x = x.clone()
            if debug:
                print(chunk_size)
            while _x.size(1) > 0:
                chunk = _x[:, :chunk_size, :]
                y.append(super().forward(chunk))
                _x = _x[:, chunk_size:, :]
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


def train(rank, a, h):
    if a.device == 'xpu':
        import intel_extension_for_pytorch as ipex
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.manual_seed(h.seed)
    device = torch.device(f'{a.device}:{rank}' if rank == 0 else 'cpu')

    #generator = Generator(h).to(device)
    #generator = MySpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    generator = MySpeechT5HifiGan(h).to(device)
    if not a.generator_only:
        mpd = MultiPeriodDiscriminator().to(device)
        msd = MultiScaleDiscriminator().to(device)
    else:
        mpd = None
        msd = None

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        generator = MySpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        #generator = torch.compile(generator
        if mpd:
            mpd.load_state_dict(state_dict_do['mpd'])
        if msd:
            msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        if mpd:
            mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        if msd:
            msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    if mpd:
        assert msd is not None
        optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        if mpd:
            optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    if mpd:
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    mso_ref = MyMSO(h, a)
    mso_loss = TrainingMSO(h, a)
    trainset = MelDataset(training_filelist, h.segment_size, h.hop_size, h.sampling_rate,
                          mso_ref, mso_loss, n_cache_reuse=1024,
                          shuffle=False if h.num_gpus > 1 else True, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              prefetch_factor=2,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True,
                              persistent_workers=True if h.num_workers > 0 else False)

    if a.precompute_mels:
        for i, batch in enumerate(train_loader):
            print(f'{i}: {len(batch) * len(batch[0])}')
        exit(0)

    if rank == 0:
        validset = MelDataset(validation_filelist, h.segment_size, h.hop_size, h.sampling_rate,
                                mso_ref, mso_loss, split=False, shuffle=False, n_cache_reuse=1024,
                                device=device, fine_tuning=a.fine_tuning,
                                base_mels_path=a.input_mels_dir)
        nw = min(1, h.num_workers)
        validation_loader = DataLoader(validset, num_workers=nw, shuffle=False,
                                       prefetch_factor=2,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True,
                                       persistent_workers=True if nw > 0 else False)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    #torch.autograd.set_detect_anomaly(True)
    generator.train()
    if mpd:
        mpd.train()
    if msd:
        msd.train()

    if a.focus_mels:
        mel_weight = torch.linspace(1.0, 0.01, h.num_mels*1)
        mel_weight = mel_weight[None, :, None].to(device)

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, fn, y_mel = batch
            x = x.to(device)
            y = y.to(device)
            y_mel[0] = y_mel[0].to(device)
            if a.focus_mels:
                #print(y_mel[0].size(), y_mel[1].size(), mel_weight.size())
                y_mel[0] *=  mel_weight

            y = y.unsqueeze(1)

            y_g_hats = generator(x)
            y_g_hat_mels = [mel_spectrogram(ygh.squeeze(1), mso_loss)
                            for ygh in y_g_hats]
            if a.focus_mels:
                y_g_hat_mel[0] = y_g_hat_mel[0] * mel_weight

            # MPD
            if mpd:
                optim_d.zero_grad()
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            if msd:
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            if msd:
                loss_disc_all = loss_disc_s
            if mpd:
                loss_disc_all += loss_disc_f

            if mpd:
                loss_disc_all.backward()
                optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss

            y_mel_a = y_mel[0].repeat(len(y_g_hat_mels), 1, 1)
            y_g_hat_mels_a = torch.cat([yghm[0] for yghm in y_g_hat_mels],
                                           dim=0)
            loss_mel = F.l1_loss(y_mel_a, y_g_hat_mels_a) / 12.7
            assert not y_g_hat_mels[0][0].isnan().any() and not loss_mel.isnan().any()
            #print(f'y_g_hat_mel[0].size() = {y_g_hat_mel[0].size()}')
            #loss_mel += F.l1_loss(y_mel[1], y_g_hat_mel[1]) * 10

            if mpd:
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            if msd:
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            if mpd:
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            if msd:
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            if mpd:
                loss_gen_f, _ = generator_loss(y_df_hat_g)
            if msd:
                loss_gen_s, _ = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_mel
            if mpd:
                loss_gen_all += loss_fm_f + loss_gen_f
            if msd:
                loss_gen_all += loss_fm_s + loss_gen_s

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                stdur = time.time() - start_b
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = loss_mel.item()
                        #mel_error += F.l1_loss(y_mel[1], y_g_hat_mel[1]).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error, stdur))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    cp_data = {}
                    if mpd:
                        mpd_s = (mpd.module if h.num_gpus > 1
                                 else mpd).state_dict()
                        cp_data['mpd'] = mpd_s
                        cp_data['optim_d'] = optim_d.state_dict()
                    if msd:
                        cp_data['msd'] = (msd.module if h.num_gpus > 1
                                      else msd).state_dict()
                    cp_data['optim_g'] = optim_g.state_dict()
                    cp_data['steps'] = steps
                    cp_data['epoch'] = epoch

                    save_checkpoint(checkpoint_path, cp_data)

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    sw.add_scalar("training/secs_per_batch", stdur, steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    #torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, fn, y_mel = batch
                            fn = os.path.basename(fn[0])
                            chunksz = min(2 ** j, 32)
                            y_g_hat = generator(x.to(device), chunks=(chunksz,))[0]
                            y_mel[0] = y_mel[0].to(device)
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), mso_loss)
                            val_err_tot += F.l1_loss(y_mel[0], y_g_hat_mel[0]).item()

                            #val_err_tot += F.l1_loss(y_mel[1], y_g_hat_mel[1]).item()

                            if j <= 4:
                                loss_per_band = torch.abs(y_mel[0] - y_g_hat_mel[0]).mean(dim=1)
                                loss_per_band_np = loss_per_band.cpu().numpy()
                                sw.add_text(f'input/filename_{j}', fn, steps)
                                sw.add_audio(f'gt/y_{j}', y[0], steps, h.sampling_rate),
                                sw.add_figure(f'gt/y_spec_{j}', plot_spectrogram(y_mel[0][0].cpu().numpy()),
                                              steps)

                                sw.add_audio(f'generated/y_hat_{j}', y_g_hat[0], steps,
                                             h.sampling_rate)
                                gx_name = f'input/generator_x_{j}'
                                gx_data = x.squeeze(0).cpu().numpy()
                                gx_spec = plot_spectrogram(gx_data)
                                sw.add_figure(gx_name, gx_spec, steps)
                                mso = MyMSO(h, a)
                                mso.return_phase = False
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), mso)
                                yhs_name = f'generated/y_hat_spec_{j}'
                                yhs_data = y_hat_spec.squeeze(0).cpu().numpy()
                                yhs_spec = plot_spectrogram(yhs_data)
                                sw.add_figure(yhs_name, yhs_spec, steps)

                        val_err = val_err_tot / (j+1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        if mpd:
            scheduler_d.step()
        
        if rank == 0:
            epdur = time.time() - start
            sw.add_scalar("training/secs_per_epoch", epdur, epoch)
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(epdur)))


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
    parser.add_argument('--focus_mels', default=False, type=bool)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mel_oversample', default=1, type=int)

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

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        #multiprocessing.set_start_method('spawn', force=True)
        train(0, a, h)


if __name__ == '__main__':
    main()
