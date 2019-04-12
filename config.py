# -*- coding: utf-8 -*-
# @Time    : 18-11-16 上午11:03
# @Author  : zhoujun
import keys

trainfile = '/data1/zj/data/crnn/all/train.txt'
testfile = '/data1/zj/data/crnn/all_test/test.txt'
output_dir = 'output/resnet_all_cnndecoder'

gpu_id = '3'
workers = 6
start_epoch = 0
epochs = 100

train_batch_size = 128
eval_batch_size = 64
# img shape
img_h = 32
img_w = 320
img_channel = 3
img_type = 'cv'
nh = 512

lr = 0.001
end_lr = 1e-7
lr_decay = 0.1
lr_decay_step = 15
alphabet = keys.all_alphabet
display_interval = 100
restart_training = True
checkpoint = 'output/resnet_all/best_0.7660778739579008.pth'

# random seed
seed = 2