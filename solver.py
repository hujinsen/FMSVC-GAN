# from model import Generator
# from model import Discriminator

from model_new import Generator, Discriminator

from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import os
from os.path import join, basename, dirname, split
import time
import datetime
from data_loader import to_categorical
import librosa
from utils import *
from tqdm import tqdm
import soundfile as sf


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun

def get_scheduler(optimizer, hp, it=-1):
    if 'lr_policy' not in hp or hp['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hp['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hp['step_size'],
                                        gamma=hp['gamma'], last_epoch=it)
    else:
        return NotImplementedError('%s not implemented', hp['lr_policy'])
    return scheduler

class Solver(nn.Module):
    """Solver for training and testing StarGAN."""

    def __init__(self, train_loader, test_loader, config):
        """Initialize configurations."""
        super(Solver, self).__init__()
        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sampling_rate = config.sampling_rate

        # Model configurations.
        self.num_speakers = config.num_speakers
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""

        config = get_config('config.yaml')
        lr_gen = config['lr_gen']
        lr_dis = config['lr_dis']
        self.G = Generator(1, config['gen'])
        self.D = Discriminator(input_size=(128,256))

        # dis_params = list(self.G.parameters())
        # gen_params = list(self.D.parameters())

        # self.d_optimizer = torch.optim.RMSprop(
        #     [p for p in dis_params if p.requires_grad],
        #     lr=lr_dis, weight_decay=config['weight_decay'])

        # self.g_optimizer = torch.optim.RMSprop(
        #     [p for p in gen_params if p.requires_grad],
        #     lr=lr_gen, weight_decay=config['weight_decay'])

        # self.d_optimizer = torch.optim.Adam(
        #     [p for p in dis_params if p.requires_grad],
        #     lr=lr_dis, weight_decay=config['weight_decay'])

        # self.g_optimizer = torch.optim.Adam(
        #     [p for p in gen_params if p.requires_grad],
        #     lr=lr_gen, weight_decay=config['weight_decay'])

        # self.dis_scheduler = get_scheduler(self.d_optimizer, config)
        # self.gen_scheduler = get_scheduler(self.g_optimizer, config)
        # self.apply(weights_init(config['init']))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        
        
        
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def sample_spk_c(self, size):
        spk_c = np.random.randint(0, self.num_speakers, size=size)
        spk_c_cat = to_categorical(spk_c, self.num_speakers)
        return torch.LongTensor(spk_c), torch.FloatTensor(spk_c_cat)

    def classification_loss(self, logit, target):
        """Compute softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def load_wav(self, wavfile, sr=16000):
        # wav, _ = librosa.load(wavfile, sr=sr, mono=True)
        wav, _ = sf.read(wavfile)      
        return wav_padding(wav, sr=16000, frame_period=5, multiple = 4)  # TODO

    def train(self):
       
        # Set data loader.
        train_loader = self.train_loader

        data_iter = iter(train_loader)

        # Read a batch of testdata
        test_wavfiles = self.test_loader.get_batch_test_data(batch_size=8)
        test_wavs = [self.load_wav(wavfile) for wavfile in test_wavfiles]

        # Determine whether do copysynthesize when first do training-time conversion test.
        cpsyn_flag = [True, False][0]
        # f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print("resuming step %d ..."% self.resume_iters)
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch labels.
            try:
                mc_real, spk_label_org, spk_c_org = next(data_iter)
            except:
                data_iter = iter(train_loader)
                mc_real, spk_label_org, spk_c_org = next(data_iter)

            mc_real.unsqueeze_(1) # (B, D, T) -> (B, 1, D, T) for conv2d
            #给数据添加些噪音
            mc_real += torch.randn(mc_real.size())
            # Generate target domain labels randomly.
            # spk_label_trg: int,   spk_c_trg:one-hot representation 
            spk_label_trg, spk_c_trg = self.sample_spk_c(mc_real.size(0)) 

            mc_real = mc_real.to(self.device)                         # Input mc.
            spk_label_org = spk_label_org.to(self.device)             # Original spk labels.
            spk_c_org = spk_c_org.to(self.device)                     # Original spk acc conditioning.
            spk_label_trg = spk_label_trg.to(self.device)             # Target spk labels for classification loss for G.
            spk_c_trg = spk_c_trg.to(self.device)                     # Target spk conditioning.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # lsgan loss
            r_out = self.D(mc_real)
            # print(r_out.shape)
            mc_fake = self.G(mc_real)
            f_out = self.D(mc_fake.detach())

            # d_loss = torch.mean((f_out - 0)**2) + torch.mean((r_out - 1)**2)

            all0 = Variable(torch.zeros_like(f_out.data).cuda(), requires_grad=False)
            all1 = Variable(torch.ones_like(r_out.data).cuda(), requires_grad=False)
            d_loss = torch.mean(F.binary_cross_entropy(F.sigmoid(f_out), all0) +
                                   F.binary_cross_entropy(F.sigmoid(r_out), all1))
           
            # Compute loss for gradient penalty.
            alpha = torch.rand(mc_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * mc_real.data + (1 - alpha) * mc_fake.data).requires_grad_(True)
            out_src = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss'] = d_loss.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # lsgan loss
                mc_fake = self.G(mc_real)
                g_f_out = self.D(mc_fake)

                
                g_loss_fake = -torch.mean(g_f_out)

                #reconstruction loss
                g_loss_rec = torch.mean(torch.abs(mc_real - mc_fake))

                # Backward and optimize.
                g_loss = self.lambda_rec * g_loss_rec
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/g_loss'] = g_loss.item()
            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)
            
            if (i+1) % self.sample_step == 0:
                sampling_rate=16000
                num_mcep=128
                frame_period=5
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
                            mean_log_src=self.test_loader.logf0s_mean_src, std_log_src=self.test_loader.logf0s_std_src, 
                            mean_log_target=self.test_loader.logf0s_mean_trg, std_log_target=self.test_loader.logf0s_std_trg)
                        coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
                        coded_sp_norm = (coded_sp - self.test_loader.mcep_mean_src) / self.test_loader.mcep_std_src
                        coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1)
                        
                        _, _, sp_trg, _ = world_decompose(wav=wav_trg, fs=sampling_rate, frame_period=frame_period)
                        coded_sp_trg = world_encode_spectral_envelop(sp=sp_trg, fs=sampling_rate, dim=num_mcep)
                        coded_sp_norm_trg = (coded_sp_trg - self.test_loader.mcep_mean_trg) / self.test_loader.mcep_std_trg
                        coded_sp_norm_tensor_trg = torch.FloatTensor(coded_sp_norm_trg.T).unsqueeze_(0).unsqueeze_(1)
                        #目标说话人音频
                        new_src_frames = make_frames(coded_sp_norm_tensor)
                        new_trg_frames = make_frames(coded_sp_norm_tensor_trg)
                       
                        new_src, new_trg = make_same_batch(new_src_frames, new_trg_frames)

                        new_src = new_src.to(self.device)
                        new_trg = new_trg.to(self.device)

                        coded_sp_converted_norm = self.G.convert(new_src, new_trg).data.cpu().numpy()

                        # coded_sp_converted = np.squeeze(coded_sp_converted_norm).T * self.test_loader.mcep_std_trg + self.test_loader.mcep_mean_trg
                        # coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                        # decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)

                        res = []
                        for i in range(coded_sp_converted_norm.shape[0]):
                            coded_sp_converted = np.squeeze(coded_sp_converted_norm[i]).T * self.test_loader.mcep_std_trg + self.test_loader.mcep_mean_trg
                            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                            wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted, 
                                                                    ap=ap, fs=sampling_rate, frame_period=frame_period)
                            res.extend(wav_transformed)

                        sf.write(
                            join(self.sample_dir, str(i+1)+'-'+wav_name_src.split('.')[0]+'-vcto-{}'.format(wav_name_trg.split('.')[0])+'.wav'), res[:len(wav_src)], sampling_rate)
                        
                        if cpsyn_flag:
                            wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp, 
                                                        ap=ap, fs=sampling_rate, frame_period=frame_period)
                            # librosa.output.write_wav(join(self.sample_dir, 'cpsyn-'+wav_name), wav_cpsyn, sampling_rate)

                            sf.write(join(self.sample_dir, 'cpsyn-'+wav_name_src), wav_cpsyn, sampling_rate)
                    cpsyn_flag = False

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


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