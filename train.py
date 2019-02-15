# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:20
# @Author  : zhoujun

from __future__ import print_function
import os
import copy
import time
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data
from torch import nn
from torch.nn import CTCLoss
import utils
from crnn_mx import CRNN
import config
import shutil
from dataset import ImageDataset
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
if config.img_type == 'cv':
    from opencv_transforms import opencv_transforms as transforms
else:
    from torchvision import transforms

def setup_logger(log_file_path: str = None):
    import logging
    from colorlog import ColoredFormatter
    logging.basicConfig(filename=log_file_path, format='%(asctime)s %(levelname)-8s %(filename)s: %(message)s',
                        # 定义输出log的格式
                        datefmt='%Y-%m-%d %H:%M:%S', )
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter("%(asctime)s %(log_color)s%(levelname)-8s %(reset)s %(filename)s: %(message)s",
                                 datefmt='%Y-%m-%d %H:%M:%S',
                                 reset=True,
                                 log_colors={
                                     'DEBUG': 'blue',
                                     'INFO': 'green',
                                     'WARNING': 'yellow',
                                     'ERROR': 'red',
                                     'CRITICAL': 'red',
                                 })

    logger = logging.getLogger('project')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('logger init finished')
    return logger


def save_checkpoint(checkpoint_path, model, optimizer, epoch, logger):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch}
    torch.save(state, checkpoint_path)
    logger.info('model saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, logger):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
    logger.info('model loaded from %s' % checkpoint_path)
    return start_epoch


def accuracy(preds, labels, preds_lengths, converter):
    _, preds = preds.max(2)
    preds = preds.squeeze(1)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(preds.data, preds_lengths.data, raw=False)
    n_correct = 0
    for pred, target in zip(sim_preds, labels):
        if pred == target:
            n_correct += 1
    return n_correct


def evaluate_accuracy(model, dataloader, device, converter):
    model.eval()
    metric = 0
    pbar = tqdm(total=len(dataloader),desc='test crnn')
    for i, (images, label) in enumerate(dataloader):
        cur_batch_size = images.size(0)
        with torch.no_grad():
            images = images.to(device)
        preds = model(images)
        # print(len(images))
        preds_lengths = torch.Tensor([preds.size(0)] * cur_batch_size).int()
        metric += accuracy(preds.cpu(), label, preds_lengths.cpu(), converter)
        pbar.update(1)
    pbar.close()
    return metric


# custom weights initialization called on crnn
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=-0.07, b=0.07)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def train():
    torch.random.initial_seed()
    if config.restart_training:
        shutil.rmtree(config.output_dir, ignore_errors=True)
    if config.output_dir is None:
        config.output_dir = 'output'
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    logger = setup_logger(os.path.join(config.output_dir, 'train_log'))

    # torch.manual_seed(config.seed)  # 为CPU设置随机种子
    if config.gpu_id is not None and torch.cuda.is_available():
        logger.info('train with gpu {} and pytorch {}'.format(config.gpu_id, torch.__version__))
        device = torch.device("cuda:0")
        # torch.cuda.manual_seed(config.seed)  # 为当前GPU设置随机种子
        # torch.cuda.manual_seed_all(config.seed)  # 为所有GPU设置随机种子
    else:
        logger.info('train with cpu and pytorch {}'.format(torch.__version__))
        device = torch.device("cpu")

    train_transfroms = transforms.Compose([
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor()
    ])

    train_dataset = ImageDataset(data_txt=config.trainfile, data_shape=(config.img_h, config.img_w),
                                 img_type=config.img_type, img_channel=config.img_channel, phase='train',
                                 transform=train_transfroms)

    train_data_loader = DataLoader(train_dataset, config.train_batch_size, shuffle=True, num_workers=config.workers)

    test_dataset = ImageDataset(data_txt=config.testfile, data_shape=(config.img_h, config.img_w),
                                img_type=config.img_type, img_channel=config.img_channel, phase='test',
                                transform=transforms.ToTensor())
    test_data_loader = DataLoader(test_dataset, config.eval_batch_size, shuffle=True, num_workers=config.workers)

    logger.info('load data finish')
    converter = utils.strLabelConverter(config.alphabet)
    criterion = CTCLoss()

    model = CRNN(config.img_channel, len(config.alphabet), config.nh)
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)


    start_epoch = config.start_epoch
    if config.checkpoint != '' and not config.restart_training:
        start_epoch = load_checkpoint(config.checkpoint, model, optimizer, logger)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_decay_step, gamma=config.lr_decay,last_epoch=start_epoch)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_decay_step, gamma=config.lr_decay)
    model = model.to(device)
    # print(model)
    writer = SummaryWriter(config.output_dir)
    # dummy_input = torch.Tensor(1, config.img_channel, config.img_h, config.img_w).to(device)
    # writer.add_graph(model=model, input_to_model=dummy_input)
    all_step = len(train_data_loader)
    best_acc = 0
    best_model = None
    try:
        for epoch in range(start_epoch + 1, config.epochs):
            model.train()
            if float(scheduler.get_lr()[0]) > config.end_lr:
                scheduler.step()
            start = time.time()

            batch_acc = .0
            batch_loss = .0
            cur_step = 0
            for i, (images, labels) in enumerate(train_data_loader):
                cur_batch_size = images.size(0)
                targets, targets_lengths = converter.encode(labels)

                targets = torch.Tensor(targets).int()
                targets_lengths = torch.Tensor(targets_lengths).int()
                images = images.to(device)

                preds = model(images)
                preds = preds.log_softmax(2) # for torch 1.0
                preds_lengths = torch.Tensor([preds.size(0)] * cur_batch_size).int()
                loss = criterion(preds, targets, preds_lengths, targets_lengths)  # text,preds_size must be cpu
                # backward
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()

                loss = loss.item() / cur_batch_size
                acc = accuracy(preds.cpu(), labels, preds_lengths.cpu(), converter) / cur_batch_size
                batch_acc += acc
                batch_loss += loss
                # write tensorboard
                cur_step = epoch * all_step + i
                writer.add_scalar(tag='ctc_loss', scalar_value=loss, global_step=cur_step)
                writer.add_scalar(tag='train_acc', scalar_value=acc, global_step=cur_step)
                writer.add_scalar(tag='lr', scalar_value=scheduler.get_lr()[0], global_step=cur_step)

                if (i + 1) % config.display_interval == 0:
                    batch_time = time.time() - start
                    # for name, param in model.named_parameters():
                    #     if 'bn' not in name:
                    #         writer.add_histogram(name, param, cur_step)
                    #         writer.add_histogram(name + '-grad', param.grad, cur_step)

                    logger.info(
                        '[{}/{}], [{}/{}],step: {}, Speed: {:.3f} samples/sec, ctc loss:{:.4f}, acc:{:.4f}, lr:{}, '
                        'time:{:.4f}'.format(epoch, config.epochs, i + 1, all_step, cur_step,
                                             config.display_interval * config.train_batch_size / batch_time,
                                             batch_loss / config.display_interval,
                                             batch_acc / config.display_interval,
                                             scheduler.get_lr()[0], batch_time))
                    batch_loss = .0
                    batch_acc = .0
                    start = time.time()
            logger.info('start eval....')
            # test
            val_acc = evaluate_accuracy(model, test_data_loader, device, converter) / len(test_dataset)
            logger.info('[{}/{}], val_acc: {:.6f}'.format(epoch, config.epochs, val_acc))
            writer.add_scalar(tag='val_acc', scalar_value=val_acc, global_step=cur_step)
            save_checkpoint('{}/{}_{}.pth'.format(config.output_dir, epoch, val_acc), model, optimizer, epoch,
                            logger)
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = copy.deepcopy(model)
        logger.info('best_acc is {}'.format(best_acc))
        save_checkpoint('{}/best_{}.pth'.format(config.output_dir, best_acc), best_model, optimizer, config.epochs + 1,
                        logger)
    except KeyboardInterrupt:
        logger.info('best_acc is {}'.format(best_acc))
        save_checkpoint('{}/best_{}.pth'.format(config.output_dir, best_acc), best_model, optimizer, config.epochs + 1,
                        logger)

if __name__ == '__main__':
    train()
