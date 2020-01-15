import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_loader import get_loader

#========================================================
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='reflect'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class ContentEncoder(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=6):
        super(ContentEncoder, self).__init__()

        layers = []
        
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=(3, 9), padding=(1, 4), bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        
        layers.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim*2, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim*4, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        self.downsampling = nn.Sequential(*layers)

        middle = []
        curr_dim = conv_dim * 4
        # Bottleneck layers.
        for i in range(repeat_num):
            middle.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.bottle_neck = nn.Sequential(*middle)
        
        self.__setattr__('output_dim', curr_dim)

    def forward(self, x):
        
        # print(f' x shape:{x.shape}')
        x = self.downsampling(x)
        # print(f'x after downsampling: {x.shape}')

        x = self.bottle_neck(x)
        # print(f'x after bottle_neck: {x.shape}')

        return x


class SpeakerEncoder(nn.Module):
    def __init__(self, conv_dim=64, style_dim=256,  repeat_num=6):
        super(SpeakerEncoder, self).__init__()

        layers = []
        
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=(4, 8), padding=(1, 4), bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        
        layers.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(conv_dim*4, conv_dim*4, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False))
        layers.append(nn.ReLU(inplace=True))

        layers += [nn.AdaptiveAvgPool2d(1)]
        layers += [nn.Conv2d(conv_dim * 4, style_dim, 1, 1, 0)]
        self.downsampling = nn.Sequential(*layers)


    def forward(self, x):
        
        # print(f' x shape:{x.shape}')
        x = self.downsampling(x)
        # print(f'x after downsampling: {x.shape}')

        # x = self.bottle_neck(x)
        # print(f'x after bottle_neck: {x.shape}')

        return x



class Decoder(nn.Module):
    def __init__(self,  input_dim, output_dim, res_norm='adain', activ='selu', pad_type='reflect'):
        super(Decoder, self).__init__()

       
        # upsampling blocks

        curr_dim = input_dim

        self.seq1 = nn.Sequential(
            nn.ConvTranspose2d(input_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNorm(curr_dim // 2),
            nn.ReLU()
        )

        curr_dim = curr_dim//2
        self.seq2 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNorm(curr_dim // 2),
            nn.ReLU()
        )

         # AdaIN residual blocks
        self.model = nn.Sequential(
            # *ResBlocks(5, input_dim, res_norm, activ, pad_type=pad_type)
             ResBlock( curr_dim//2, norm=res_norm, activation=activ, pad_type=pad_type),
             
             ResBlock( curr_dim//2, norm=res_norm, activation=activ, pad_type=pad_type),
             nn.Dropout2d(0.2),
             ResBlock( curr_dim//2, norm=res_norm, activation=activ, pad_type=pad_type),
             nn.Dropout2d(0.3),
             ResBlock( curr_dim//2, norm=res_norm, activation=activ, pad_type=pad_type),
             nn.Dropout2d(0.4),
             ResBlock( curr_dim//2, norm=res_norm, activation=activ, pad_type=pad_type),
             nn.Dropout2d(0.2)
        ) 
    
        self.seq3 = nn.Sequential(
           nn.Conv2d(curr_dim//2, 1, kernel_size=7, stride=1, padding=3, bias=False),
           nn.Tanh()
       )
       
        
        self.model = nn.Sequential(*self.seq1, *self.seq2, *self.model,  *self.seq3)
        
      
    def forward(self, x):
        x = self.model(x)
        return x


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64):
        super(Generator, self).__init__()
        style_dim = 128
        self.content = ContentEncoder(conv_dim)
        self.speaker = SpeakerEncoder(conv_dim, style_dim)

        self.decoder = Decoder(input_dim=self.content.output_dim, output_dim=1,  res_norm='adain', activ='selu', pad_type='reflect')
        # MLP to generate AdaIN parameters
        # self.mlp = MLP(style_dim, self.get_num_adain_params(self.decoder), 256, 3, norm='none', activ='selu')
         self.mlp = MLP(style_dim,
                       get_num_adain_params(self.decoder),
                       256,
                       3,
                       norm='none',
                       activ='relu')
    def forward(self, x):
        
        content, spk_info = self.encode(x)
        
        out = self.decode(content, spk_info)

        return out
    
    def encode(self, x):
        style = self.speaker(x)
        content = self.content(x)
        return content, style

    def decode(self, content, style):
        adain_params = self.mlp(style)
        assign_adain_params(adain_params, self.decoder)
        new_content = self.decoder(content)
        return new_content

    # def assign_adain_params(self, adain_params, model):
    #     # assign the adain_params to the AdaIN layers in model
    #     for m in model.modules():
    #         if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
    #             mean = adain_params[:, :m.num_features]
    #             std = adain_params[:, m.num_features:2*m.num_features]
    #             m.bias = mean.contiguous().view(-1)
    #             m.weight = std.contiguous().view(-1)
    #             if adain_params.size(1) > 2*m.num_features:
    #                 adain_params = adain_params[:, 2*m.num_features:]

    # def get_num_adain_params(self, model):
    #     # return the number of AdaIN parameters needed by the model
    #     num_adain_params = 0
    #     for m in model.modules():
    #         if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
    #             num_adain_params += 2 * m.num_features
    #     return num_adain_params

    def convert(self, src, trg):
        src_content = self.content(src) #
        trg_speaker = self.speaker(trg) #
      
        adain_params = self.mlp(trg_speaker)
        self.assign_adain_params(adain_params, self.decoder)
        new_content = self.decoder(src_content)

        return new_content

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, input_size=(36, 256), conv_dim=64, repeat_num=5):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout2d(0.3))
            curr_dim = curr_dim * 2
        
        kernel_size_0 = int(input_size[0] / np.power(2, repeat_num)) # 1
        kernel_size_1 = int(input_size[1] / np.power(2, repeat_num)) # 8
        self.main = nn.Sequential(*layers)
        self.conv_dis = nn.Conv2d(curr_dim, 1, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False) # padding should be 0
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv_dis(h)
    
        return out_src


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))




##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def test_content_encoder():
    ce = ContentEncoder()
    d = torch.randn((2, 1, 36, 256))
    ce_out = ce(d)
    print(ce_out.shape)


def test_speaker_encoder():
    ce = SpeakerEncoder()
    d = torch.randn((1, 1, 36, 256))
    ce_out = ce(d)
    print(ce_out.shape)


def test_decoder():
    # dc = Decoder(input_dim=256, output_dim=1)
    dc = Decoder(2,2,16,1)
    # d = torch.randn((2,512,36,256))
    d = torch.randn((1, 16, 112, 112))
    # d1 = ContentEncoder()(d)
    # d2 = SpeakerEncoder()(d)

    # d3  = torch.cat((d1,d2), dim=1)
    # print(d3.shape)

    dc_out = dc(d)
    print(dc_out.shape)

def test_generator():
    g = Generator()
    d = torch.randn((2,1,128,128))
    g_out = g(d)
    print(g_out.shape)

    dis = Discriminator(input_size=(128,128))(g_out)
    print(dis.shape)

if __name__ == '__main__':
    # test_speaker_encoder()

    test_generator()


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_loader = get_loader('/home/hu/disk1/EXP_Part2/exp_part2_2/data/mc/test', 4, 'train', num_workers=1)
    # data_iter = iter(train_loader)
    # G = Generator().to(device)
    # D = Discriminator().to(device)
    # for i in range(10):
    #     mc_real, spk_label_org, spk_label_onehot = next(data_iter)
    #     mc_real.unsqueeze_(1) # (B, D, T) -> (B, 1, D, T) for conv2d
    #     mc_real = mc_real.to(device)                         # Input mc.
    #     spk_label_org = spk_label_org.to(device)            # Original spk labels.
    #     spk_label_onehot = spk_label_onehot.to(device)             

    #     mc_fake = G(mc_real)
    #     print(mc_fake.size())

    #     out_src = D(mc_fake)
    #     print(out_src.shape)



