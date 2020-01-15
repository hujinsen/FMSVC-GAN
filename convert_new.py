import argparse
from model_new import Generator
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import os
from os.path import join, basename, dirname, split
import time
import datetime
from data_loader import to_categorical
import librosa
from utils import *
import glob
import soundfile as sf

# Below is the accent info for the used 10 speakers.
spk2acc = {'262': 'Edinburgh', #F
           '272': 'Edinburgh', #M
           '229': 'SouthEngland', #F 
           '232': 'SouthEngland', #M
           '292': 'NorthernIrishBelfast', #M 
           '293': 'NorthernIrishBelfast', #F 
           '360': 'AmericanNewJersey', #M
           '361': 'AmericanNewJersey', #F
           '248': 'India', #F
           '251': 'India'} #M
min_length = 256   # Since we slice 256 frames from each utterance when training.
# Build a dict useful when we want to get one-hot representation of speakers.
speakers = ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251']
spk2idx = dict(zip(speakers, range(len(speakers))))

class TestDataset(object):
    """Dataset for testing."""
    def __init__(self, config):
        assert config.trg_spk in speakers, f'The trg_spk should be chosen from {speakers}, but you choose {trg_spk}.'
        # Source speaker
        self.src_spk = config.src_spk
        self.trg_spk = config.trg_spk
        self.mc_files = sorted(glob.glob(join(config.test_data_dir, '{}*.npy'.format(self.src_spk))))
        self.mc_files_trg = sorted(glob.glob(join(config.test_data_dir, '{}*.npy'.format(self.trg_spk))))

        self.src_spk_stats = np.load(join(config.test_data_dir.replace('test', 'train'), '{}_stats.npz'.format(self.src_spk)))
        self.trg_spk_stats = np.load(join(config.test_data_dir.replace('test', 'train'), '{}_stats.npz'.format(self.trg_spk)))
        
        self.logf0s_mean_src = self.src_spk_stats['log_f0s_mean']
        self.logf0s_std_src = self.src_spk_stats['log_f0s_std']

        self.logf0s_mean_trg = self.trg_spk_stats['log_f0s_mean']
        self.logf0s_std_trg = self.trg_spk_stats['log_f0s_std']

        self.mcep_mean_src = self.src_spk_stats['coded_sps_mean']
        self.mcep_std_src = self.src_spk_stats['coded_sps_std']

        self.mcep_mean_trg = self.trg_spk_stats['coded_sps_mean']
        self.mcep_std_trg = self.trg_spk_stats['coded_sps_std']

        self.src_wav_dir = f'{config.wav_dir}/{self.src_spk}'
        self.trg_wav_dir = f'{config.wav_dir}/{self.trg_spk}'


    def get_batch_test_data(self, batch_size=8):
        batch_data = []
        for i in range(batch_size):
            mcfile = self.mc_files[i]
            mcfile_trg = self.mc_files_trg[i]

            filename = basename(mcfile).split('-')[-1]
            filename_trg = basename(mcfile_trg).split('-')[-1]

            wavfile_path = join(self.src_wav_dir, filename.replace('npy', 'wav'))
            wavfile_path_trg = join(self.trg_wav_dir, filename_trg.replace('npy', 'wav'))

            batch_data.append(wavfile_path)
            batch_data.append(wavfile_path_trg)

        return batch_data


def test(config):
    os.makedirs(join(config.convert_dir, str(config.resume_iters)), exist_ok=True)
    sampling_rate, num_mcep, frame_period=16000, 128, 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 # Restore model
    print(f'Loading the trained models from step {config.resume_iters}...')
    c = get_config('config.yaml')

    G = Generator(1, c['gen']).to(device)
    print(G)
    G_path = join(config.model_save_dir, f'{config.resume_iters}-G.ckpt')
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    G.eval()
    
    test_loader = TestDataset(config)
    # Read a batch of testdata
    test_wavfiles = test_loader.get_batch_test_data(batch_size=config.num_converted_wavs)
    test_wavs = [load_wav(wavfile, sampling_rate) for wavfile in test_wavfiles]

    with torch.no_grad():
        for idx in range(0, len(test_wavs)-2, 2):
            wav_src = test_wavs[idx]
            wav_trg = test_wavs[idx + 1]

            wav_name_src = basename(test_wavfiles[idx])
            wav_name_trg = basename(test_wavfiles[idx + 1])
            print(f'Source : {wav_name_src}----> Traget: {wav_name_trg}')
            #源说话人音频
            f0, timeaxis, sp, ap = world_decompose(wav=wav_src, fs=sampling_rate, frame_period=frame_period)
            f0_converted = pitch_conversion(f0=f0, 
                mean_log_src=test_loader.logf0s_mean_src, std_log_src=test_loader.logf0s_std_src, 
                mean_log_target=test_loader.logf0s_mean_trg, std_log_target=test_loader.logf0s_std_trg)
            coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_norm = (coded_sp - test_loader.mcep_mean_src) / test_loader.mcep_std_src
            coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1)
            
            _, _, sp_trg, _ = world_decompose(wav=wav_trg, fs=sampling_rate, frame_period=frame_period)
            coded_sp_trg = world_encode_spectral_envelop(sp=sp_trg, fs=sampling_rate, dim=num_mcep)
            coded_sp_norm_trg = (coded_sp_trg - test_loader.mcep_mean_trg) / test_loader.mcep_std_trg
            coded_sp_norm_tensor_trg = torch.FloatTensor(coded_sp_norm_trg.T).unsqueeze_(0).unsqueeze_(1)
            #目标说话人音频
            new_src_frames = make_frames(coded_sp_norm_tensor)
            new_trg_frames = make_frames(coded_sp_norm_tensor_trg)
            
            new_src, new_trg = make_same_batch(new_src_frames, new_trg_frames)

            new_src = new_src.to(device)
            new_trg = new_trg.to(device)

            coded_sp_converted_norm = G.convert(new_src, new_trg).data.cpu().numpy()

            # coded_sp_converted = np.squeeze(coded_sp_converted_norm).T * test_loader.mcep_std_trg + test_loader.mcep_mean_trg
            # coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            # decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)

            res = []
            for i in range(coded_sp_converted_norm.shape[0]):
                coded_sp_converted = np.squeeze(coded_sp_converted_norm[i]).T * test_loader.mcep_std_trg + test_loader.mcep_mean_trg
                coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted, 
                                                        ap=ap, fs=sampling_rate, frame_period=frame_period)
                res.extend(wav_transformed)

            sf.write(
                join(config.convert_dir, str(i+1)+'-'+wav_name_src.split('.')[0]+'-vcto-{}'.format(wav_name_trg.split('.')[0])+'.wav'), res[:len(wav_src)], sampling_rate)
            

            wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp, 
                                        ap=ap, fs=sampling_rate, frame_period=frame_period)

            sf.write(join(config.convert_dir, 'cpsyn-'+wav_name_src), wav_cpsyn, sampling_rate)
        

def make_frames(one_tensor: torch.FloatTensor, frames=256):
  
    #将36x888的矩阵切分为若干36xframes的
    lastdim = one_tensor.shape[-1]
    if lastdim % frames != 0:
        needpad = ((lastdim//frames + 1) * frames) - lastdim
        print(needpad)

    p1d = (0, needpad) # pad last dim by 1 on each side
    padded = F.pad(one_tensor, p1d, "constant", 0)  # effectively zero padding

    res = []
    for i in range(0, padded.shape[-1], frames):
        res.append(padded[:,:,:,i:i+frames])
    res = torch.cat(res, dim=0)
    
    return res

def make_same_batch(src:torch.Tensor, trg:torch.Tensor):
    #将batchsize设置为相同，转换时
    a = src.shape[0]
    b = trg.shape[0]

    if a < b:
        trg = trg[:a, :,:,:]
    elif a > b:
        z = torch.zeros((a-b, *src.shape[1:]))
        trg = torch.cat((trg, z), dim=0)

    return src, trg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Model configuration.
    parser.add_argument('--num_speakers', type=int, default=10, help='dimension of speaker labels')
    parser.add_argument('--num_converted_wavs', type=int, default=20, help='number of wavs to convert.')
    parser.add_argument('--resume_iters', type=int, default=None, help='step to resume for testing.')
    parser.add_argument('--src_spk', type=str, default='p229', help = 'target speaker.')
    parser.add_argument('--trg_spk', type=str, default='p272', help = 'target speaker.')

    # Directories.
    parser.add_argument('--train_data_dir', type=str, default='data/mc/train')
    parser.add_argument('--test_data_dir', type=str, default='data/mc/test')
    parser.add_argument('--wav_dir', type=str, default="data/VCTK-Corpus/wav16")
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--model_save_dir', type=str, default='models')
    parser.add_argument('--convert_dir', type=str, default='converted')


    config = parser.parse_args()
    
    config.resume_iters = 15000
    config.src_spk = 'p262'
    config.trg_spk = 'p232'

    print(config)
    if config.resume_iters is None:
        raise RuntimeError("Please specify the step number for resuming.")
    test(config)

# python convert.py --resume_iters 795000 --src_spk p229 --trg_spk p262