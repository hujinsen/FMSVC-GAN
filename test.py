import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

def make_frames(one_tensor: torch.FloatTensor, frames=256):
  
    #将36x888的矩阵切分为若干36xframes的
    lastdim = one_tensor.shape[-1]
    if lastdim % frames != 0:
        needpad = ((lastdim//frames + 1) * frames) - lastdim
        print(needpad)

    p1d = (needpad//2, needpad//2) # pad last dim by 1 on each side
    padded = F.pad(one_tensor, p1d, "constant", 0)  # effectively zero padding

    res = []
    for i in range(0, padded.shape[-1], frames):
        res.append(padded[:,:,:,i:i+frames])
    res = torch.cat(res, dim=0)
  
    return res

def make_same_batch(src:torch.Tensor, trg:torch.Tensor):
    a = src.shape[0]
    b = trg.shape[0]

    if a < b:
        trg = trg[:a, :,:,:]
    elif a > b:
        z = torch.zeros((a-b, *src.shape[1:]))
        trg = torch.cat((trg, z), dim=0)

    return src, trg

# m = nn.AdaptiveAvgPool2d(((1)))
# input = torch.randn(1, 64, 8, 9)
# output = m(input)
# print(output.shape)
# print(output)

if __name__=='__main__':
    # t4d = torch.empty(1, 1, 36, 784)

    # o = make_frames(t4d, frames=256)
    # print(o.shape)

    a = torch.randn((5,1,36,256))
    b = torch.randn((3,1,36,256))

    a_, b_ = make_same_batch(a,b)
    print(a_.shape, b_.shape)