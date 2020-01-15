import argparse
from model import Generator
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


def test_two_speech(source:str, target:str, resume_iters=0):

    src_spk = basename(source).split('_')[0]
    trg_spk = basename(target).split('_')[0]

    wav_name_src = basename(source)
    wav_name_trg = basename(target)

    src_spk_stats = np.load(join('data/mc/train/', f'{src_spk}_stats.npz'))
    trg_spk_stats = np.load(join('data/mc/train/', f'{trg_spk}_stats.npz'))
    src_wav_dir = dirname(source)

    logf0s_mean_src = src_spk_stats['log_f0s_mean']
    logf0s_std_src = src_spk_stats['log_f0s_std']
    logf0s_mean_trg = trg_spk_stats['log_f0s_mean']
    logf0s_std_trg = trg_spk_stats['log_f0s_std']
    mcep_mean_src = src_spk_stats['coded_sps_mean']
    mcep_std_src = src_spk_stats['coded_sps_std']
    mcep_mean_trg = trg_spk_stats['coded_sps_mean']
    mcep_std_trg = trg_spk_stats['coded_sps_std']

    outdir = join('converted/', str(resume_iters))
    os.makedirs(outdir, exist_ok=True)
    sampling_rate, num_mcep, frame_period = 16000, 36, 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = Generator().to(device)
    # Restore model models/525000-G.ckpt
    print(f'Loading the trained models from step {resume_iters}...')
    G_path = join('models/', f'{resume_iters}-G.ckpt')
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    G.eval()

    wav_src = load_wav(source, sampling_rate)
    wav_trg = load_wav(target, sampling_rate)

    with torch.no_grad():
        #源说话人音频
        f0, _, sp, ap = world_decompose(wav=wav_src, fs=sampling_rate, frame_period=frame_period)
        f0_converted = pitch_conversion(f0=f0, 
            mean_log_src=logf0s_mean_src, std_log_src=logf0s_std_src, 
            mean_log_target=logf0s_mean_trg, std_log_target=logf0s_std_trg)
        coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
        coded_sp_norm = (coded_sp - mcep_mean_src) / mcep_std_src
        coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1)
        
        _, _, sp_trg, _ = world_decompose(wav=wav_trg, fs=sampling_rate, frame_period=frame_period)
        coded_sp_trg = world_encode_spectral_envelop(sp=sp_trg, fs=sampling_rate, dim=num_mcep)
        coded_sp_norm_trg = (coded_sp_trg - mcep_mean_trg) / mcep_std_trg
        coded_sp_norm_tensor_trg = torch.FloatTensor(coded_sp_norm_trg.T).unsqueeze_(0).unsqueeze_(1)
        #目标说话人音频
        new_src_frames = make_frames(coded_sp_norm_tensor)
        new_trg_frames = make_frames(coded_sp_norm_tensor_trg)
        
        new_src, new_trg = make_same_batch(new_src_frames, new_trg_frames)

        new_src = new_src.to(device)
        new_trg = new_trg.to(device)

        coded_sp_converted_norm = G.convert(new_src, new_trg).data.cpu().numpy()

        res = []
        for i in range(coded_sp_converted_norm.shape[0]):
            coded_sp_converted = np.squeeze(coded_sp_converted_norm[i]).T * mcep_std_trg + mcep_mean_trg
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted, 
                                                    ap=ap, fs=sampling_rate, frame_period=frame_period)
            res.extend(wav_transformed)

        sf.write(
            join(outdir, wav_name_src.split('.')[0]+'-vcto-{}'.format(wav_name_trg.split('.')[0])+'.wav'), res[:len(wav_src)], sampling_rate)
        
        
        wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp, 
                                    ap=ap, fs=sampling_rate, frame_period=frame_period)

        sf.write(join(outdir, 'cpsyn-'+wav_name_src), wav_cpsyn, sampling_rate)


def test_two_speech_custom_normalize(source:str, target:str, resume_iters=0):
#不使用提前计算的均值方差，使用自身的均值方差归一化
    src_spk = basename(source).split('_')[0]
    trg_spk = basename(target).split('_')[0]

    wav_name_src = basename(source)
    wav_name_trg = basename(target)

    outdir = join('converted/', str(resume_iters))
    os.makedirs(outdir, exist_ok=True)
    sampling_rate, num_mcep, frame_period = 16000, 36, 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = Generator().to(device)
    # Restore model models/525000-G.ckpt
    print(f'Loading the trained models from step {resume_iters}...')
    G_path = join('models/', f'{resume_iters}-G.ckpt')
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    G.eval()

    wav_src = load_wav(source, sampling_rate)
    wav_trg = load_wav(target, sampling_rate)

    with torch.no_grad():
        #源说话人音频
        f0, _, sp, ap = world_decompose(wav=wav_src, fs=sampling_rate, frame_period=frame_period)
        logf0s_mean_src, logf0s_std_src = f0.mean(), f0.std()

        coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
        mcep_mean_src, mcep_std_src = coded_sp_stat(coded_sp)

        coded_sp_norm = (coded_sp - mcep_mean_src) / mcep_std_src
        coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1)
        
        f0_t, _, sp_trg, _ = world_decompose(wav=wav_trg, fs=sampling_rate, frame_period=frame_period)
        logf0s_mean_trg, logf0s_std_trg = f0_t.mean(), f0_t.std()

        coded_sp_trg = world_encode_spectral_envelop(sp=sp_trg, fs=sampling_rate, dim=num_mcep)
        mcep_mean_trg, mcep_std_trg = coded_sp_stat(coded_sp_trg)

        coded_sp_norm_trg = ((coded_sp_trg - mcep_mean_trg) / mcep_std_trg)

        coded_sp_norm_tensor_trg = torch.FloatTensor(coded_sp_norm_trg.T).unsqueeze_(0).unsqueeze_(1)
        #f0转换
        f0_converted = pitch_conversion(f0=f0, 
            mean_log_src=logf0s_mean_src, std_log_src=logf0s_std_src, 
            mean_log_target=logf0s_mean_trg, std_log_target=logf0s_std_trg)
        #目标说话人音频
        new_src_frames = make_frames(coded_sp_norm_tensor)
        new_trg_frames = make_frames(coded_sp_norm_tensor_trg)
        
        new_src, new_trg = make_same_batch(new_src_frames, new_trg_frames)

        new_src = new_src.to(device)
        new_trg = new_trg.to(device)

        coded_sp_converted_norm = G.convert(new_src, new_trg).data.cpu().numpy()

        res = []
        for i in range(coded_sp_converted_norm.shape[0]):
            coded_sp_converted = np.squeeze(coded_sp_converted_norm[i]).T * mcep_std_trg + mcep_mean_trg
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted, 
                                                    ap=ap, fs=sampling_rate, frame_period=frame_period)
            res.extend(wav_transformed)
        
        sf.write(
            join(outdir, wav_name_src.split('.')[0]+'-vcto-{}'.format(wav_name_trg.split('.')[0])+'.wav'), res[:len(wav_src)], sampling_rate)
        
        
        wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp, 
                                    ap=ap, fs=sampling_rate, frame_period=frame_period)

        sf.write(join(outdir, 'cpsyn-'+wav_name_src), wav_cpsyn, sampling_rate)

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


if __name__ == "__main__":
    src = 'data/VCTK-Corpus/wav16/p229/p229_001.wav'
    # trg = 'data/VCTK-Corpus/wav16/p232/p232_001.wav'
    trg = 'data/VCTK-Corpus/wav16/p229/p229_001.wav'
    resume = 5000

    # test_two_speech(src, trg, resume)
    test_two_speech_custom_normalize(src, trg, resume)