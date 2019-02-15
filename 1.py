# -*- coding: utf-8 -*-
# @Time    : 12/16/18 8:51 AM
# @Author  : zhoujun

import timeit
import sys
import torch
num_labels = 10
target_length  = 30
input_length = 50
eps = 1e-5
BLANK = 0#num_labels
batch_size = 16


torch.manual_seed(5)
activations = torch.randn(input_length, batch_size, num_labels + 1)
log_probs = torch.log_softmax(activations, 2)
probs = torch.exp(log_probs)
targets = torch.randint(1, num_labels+1, (batch_size * target_length,), dtype=torch.long)
targets_2d = targets.view(batch_size, target_length)
target_lengths = torch.tensor(batch_size*[target_length])
input_lengths = torch.tensor(batch_size*[input_length])
activations = log_probs.detach()


def time_cuda_ctc_loss(grout, *args):
    torch.cuda.synchronize()
    culo, culog_alpha = torch._ctc_loss(*args)
    g, = torch.autograd.grad(culo, args[0], grout)
    torch.cuda.synchronize()

def time_cudnn_ctc_loss(groupt, *args):
    torch.cuda.synchronize()
    culo, cugra= torch._cudnn_ctc_loss(*args)
    g, = torch.autograd.grad(culo, args[0], grout)
    torch.cuda.synchronize()


a = 'cudnn'
if a == 'cuda':
    lpcu = log_probs.float().cuda().detach().requires_grad_()
    args = [lpcu, targets_2d.cuda(), input_lengths.cuda(), target_lengths.cuda(), BLANK]
    grout = lpcu.new_ones((batch_size,))
    torch.cuda.synchronize()
    print(timeit.repeat("time_cuda_ctc_loss(grout, *args)", number=1000, globals=globals()))
elif a == 'cudnn':
    lpcu = log_probs.float().cuda().detach().requires_grad_()
    args = [lpcu, targets.int(), input_lengths.int(), target_lengths.int(), BLANK, True]
    grout = lpcu.new_ones((batch_size,))
    torch.cuda.synchronize()
    print(timeit.repeat("time_cudnn_ctc_loss(grout, *args)", number=1000, globals=globals()))