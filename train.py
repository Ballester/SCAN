# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Training script"""

import os
import time
import shutil

import torch
import numpy
import numpy as np

import data
from vocab import Vocabulary, deserialize_vocab
from model import SCAN
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_xattn_t2i, shard_xattn_i2t
from torch.autograd import Variable

import logging
from tensorboardX import SummaryWriter
import torchvision.utils as vutils


import argparse


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='precomp',
                        help='{coco,f30k}_precomp')
    
    parser.add_argument('--val_data', default='precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--val_split', default='val',
                        help='split to validate results durin training')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='batch size for validation')

    parser.add_argument('--adapt_data', default='precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--adapt_split', default='train',
                        help='split to perform adaptation to')
    parser.add_argument('--adapt_batch_size', type=int, default=128,
                        help='batch size for adaptation')
    
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='./runs/runX/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/runX/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--raw_feature_norm', default="clipped_l2norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--agg_func', default="LogSumExp",
                        help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--cross_attn', default="t2i",
                        help='t2i|i2t')
    parser.add_argument('--precomp_enc_type', default="basic",
                        help='basic|weight_norm')
    parser.add_argument('--bi_gru', action='store_true',
                        help='Use bidirectional GRU.')
    parser.add_argument('--lambda_lse', default=6., type=float,
                        help='LogSumExp temp.')
    parser.add_argument('--lambda_softmax', default=9., type=float,
                        help='Attention softmax temperature.')

    ### Jonatas additional hyper ###
    parser.add_argument('--text_encoder', default='GRU',
                        help='[GRU|Conv].')
    parser.add_argument('--test_measure', default=None,
                        help='Similarity used for retrieval (None<same used for training>|cosine|order)')
    parser.add_argument('--add_data', action='store_true',
                        help='Wheter to use additional unlabeled data.')
    parser.add_argument('--log_images', action='store_true',
                        help='Wheter to use log images in tensorboard.')
    parser.add_argument('--noise', type=float, default=0.,
                        help='Ammont of noise for augmenting embeddings.')
    parser.add_argument('--dropout_noise', type=float, default=0.,
                        help='Ammont of noise for augmenting word embeddings.')

    parser.add_argument('--kwargs', type=str, nargs='+', default=None,
                        help='Additional args for the model. Usage: argument:type:value ')


    ### Mean-teacher hyperparameters ###
    parser.add_argument('--ramp_lr', action='store_true',
                        help='Use the learning rate schedule from mean-teacher')
    parser.add_argument('--initial_lr', type=float, default=0.0006,
                        help='Initial learning_rate for rampup')
    parser.add_argument('--initial_lr_rampup', type=int, default=50,
                        help='Epoch for lr rampup')
    parser.add_argument('--consistency_weight', type=float, default=20.,
                        help='consistency weight (default: 20.).')
    parser.add_argument('--consistency_alpha', type=float, default=0.99,
                        help='Consistency alpha before ema_late_epoch')
    parser.add_argument('--consistency_alpha_late', type=float, default=0.999,
                        help='Consistency alpha after ema_late_epoch')
    parser.add_argument('--consistency_rampup', type=int, default=100,
                        help='Consistency rampup epoch')
    parser.add_argument('--ema_late_epoch', type=int, default=50,
                        help='When to change alpha variable for consistency weight')

    opt = parser.parse_args()

    # if opt.test_measure is None:
    #     opt.test_measure = opt.measure

    print('\n\n')
    print(opt)


    if opt.logger_name == '':
        writer = SummaryWriter()
        logpath = writer.file_writer.get_logdir()
        opt.logger_name = logpath
    else:
        writer = SummaryWriter(opt.logger_name)

    print('')
    print('')
    print('Outpath: ', opt.logger_name)


    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    # Load Vocabulary Wrapper
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    # Load data loaders
    # train_loader, val_loader = data.get_loaders(
    #     opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    train_loader = data.get_loader(
        split='train',
        data_name=opt.data_name,
        batch_size=opt.batch_size,
        vocab=vocab,
        # tokenizer=tokenizer,        
        workers=opt.workers,
        opt=opt,
        adapt_set=False,
    )
    print('[OK] Train loader')

    val_loader = data.get_loader(
        data_name=opt.val_data,
        split=opt.val_split,
        batch_size=opt.val_batch_size,
        vocab=vocab,
        # tokenizer=tokenizer,                
        workers=opt.workers,
        opt=opt,
        adapt_set=False,
    )
    
    print('[OK] Val loader')

    adapt_loader = data.get_loader(
            split=opt.adapt_split,
            data_name=opt.adapt_data,
            batch_size=opt.adapt_batch_size,
            vocab=vocab,
            # tokenizer=tokenizer,            
            workers=opt.workers,
            opt=opt,
            adapt_set=True,            
        )
    
    print('[OK] Adapt loader')

    print('Train loader/dataset')
    print(train_loader.dataset.data_path, train_loader.dataset.split)
    print('Valid loader/dataset')
    print(val_loader.dataset.data_path, val_loader.dataset.split)
    print('Adapt loader/dataset')
    print(adapt_loader.dataset.data_path, adapt_loader.dataset.split)

    print('[OK] Loaders.')

    # Construct the model
    model = create_model(opt)
    model_ema = create_model(opt, ema=True)
    print(model.txt_enc)


    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model, writer)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0
    for epoch in range(opt.num_epochs):
        # train for one epoch
        train(opt, train_loader, adapt_loader, model, model_ema, epoch, val_loader, tb_writer=writer)

        # evaluate on validation set
        print('Validate EMA')
        rsum = validate(opt, val_loader, model, writer)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'model_ema': model_ema.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')


def train(opt, train_loader, adapt_loader, model, model_ema, epoch, val_loader, tb_writer):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    adapt_iter = iter(adapt_loader)
    adapt_loss = torch.nn.MSELoss()

    if opt.ramp_lr:
        adjust_learning_rate_mean_teacher(
            model.optimizer, epoch, opt.num_epochs,
            opt.initial_lr_rampup, opt.initial_lr)
    else:
        adjust_learning_rate(opt, model.optimizer, epoch)

    consistency_weight = get_current_consistency_weight(
        opt.consistency_weight, epoch, opt.consistency_rampup)


    for i, train_data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        model.Eiters += 1

        # switch to train mode
        model.train_start()
        model_ema.train_start()

        # make sure train logger is used
        model.logger = train_logger

        try:
            adapt_data = next(adapt_iter)
        except:
            adapt_iter = iter(adapt_loader)
            adapt_data = next(adapt_iter)

        # Get embeddings
        img_emb, cap_emb, cap_lens = model.run_emb(*train_data)        

        # Data for Domain Adaptation or SS Learning 
        # Adapt loader returns different features for the same images
        adapt_imgs_ema, adapt_imgs, adapt_caption, adapt_lens, _ = adapt_data
            
        adapt_imgs = adapt_imgs.float().cuda()                
        adapt_imgs_ema = adapt_imgs_ema.float().cuda()
        
        consistency_loss_cap = 0.
        if opt.adapt_split != 'unlabeled':            
            with torch.no_grad():
                adapt_caption = adapt_caption.cuda()
                ema_adapt_cap_emb = model_ema.txt_enc(adapt_caption, adapt_lens, dropout=opt.dropout_noise)
                adapt_cap_mb = model.txt_enc(adapt_caption, adapt_lens, dropout=opt.dropout_noise)
                consistency_loss_cap = adapt_loss(ema_adapt_cap_emb, adapt_cap_mb)
                        
        with torch.no_grad():
            ema_adapt_imgs_emb = model_ema.img_enc(adapt_imgs_ema)            

        adapt_imgs_emb = model.img_enc(adapt_imgs)

        consistency_loss_img = adapt_loss(ema_adapt_imgs_emb, adapt_imgs_emb)
        consistency_loss = (consistency_loss_img/2. + consistency_loss_cap/2.) * consistency_weight

        # measure accuracy and record loss
        model.optimizer.zero_grad()
        loss = model.forward_loss(img_emb, cap_emb, cap_lens)
        total_loss = loss + consistency_loss

        # compute gradient and do SGD step
        total_loss.backward()
        if model.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                parameters=model.params, max_norm=model.grad_clip)

        model.optimizer.step()

        if epoch <= opt.ema_late_epoch:
            update_ema_variables(
                model=model,
                ema_model=model_ema,
                alpha=opt.consistency_alpha,
                global_step=model.Eiters,
            )
        else:
            update_ema_variables(
                model=model,
                ema_model=model_ema,
                alpha=opt.consistency_alpha_late,
                global_step=model.Eiters,
            )


        # Update the model
        # model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()        

        tb_writer.add_scalar('Iter', model.Eiters, model.Eiters)
        tb_writer.add_scalar('Lr', model.optimizer.param_groups[0]['lr'], model.Eiters)
        tb_writer.add_scalar('Consistency weight', consistency_weight, model.Eiters)

        model.logger.update('Contr Loss', loss.item(), )
        model.logger.update('Adapt Loss', consistency_loss.item(), )
        model.logger.update('Total Loss', total_loss.item(), )


        # Print log info
        if model.Eiters % opt.log_step == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_writer.add_scalar('epoch', epoch, model.Eiters)
        tb_writer.add_scalar('step', i, model.Eiters)
        tb_writer.add_scalar('batch_time', batch_time.val, model.Eiters)
        tb_writer.add_scalar('data_time', data_time.val, model.Eiters)

        model.logger.tb_log(tb_writer, prefix='train', step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0 and model.Eiters > 0:
            validate(opt, val_loader, model, tb_writer)

            if opt.log_images:
                plot_img = vutils.make_grid(train_data[0],
                                normalize=True, scale_each=True)
                tb_writer.add_image('Labeled Images', plot_img, model.Eiters)

                plot_img = vutils.make_grid(adapt_imgs,
                                normalize=True, scale_each=True)
                tb_writer.add_image('Adapt Images', plot_img, model.Eiters)



def validate(opt, val_loader, model, tb_writer):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, cap_lens = encode_data(
        model, val_loader, opt.log_step, logging.info)

    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    start = time.time()
    if opt.cross_attn == 't2i':
        sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, opt, shard_size=128)
    elif opt.cross_attn == 'i2t':
        sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, opt, shard_size=128)
    else:
        raise NotImplementedError
    end = time.time()
    print("calculate similarity time:", end-start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
    print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, cap_embs, cap_lens, sims)
    print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_writer.add_scalar('data/r1', r1, model.Eiters)
    tb_writer.add_scalar('data/r5', r5, model.Eiters)
    tb_writer.add_scalar('data/r10', r10, model.Eiters)
    tb_writer.add_scalar('data/medr', medr, model.Eiters)
    tb_writer.add_scalar('data/meanr', meanr, model.Eiters)
    tb_writer.add_scalar('data/r1i', r1i, model.Eiters)
    tb_writer.add_scalar('data/r5i', r5i, model.Eiters)
    tb_writer.add_scalar('data/r10i', r10i, model.Eiters)
    tb_writer.add_scalar('data/medri', medri, model.Eiters)
    tb_writer.add_scalar('data/meanr', meanr, model.Eiters)
    tb_writer.add_scalar('data/rsum', currscore, model.Eiters)

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

### MEAN-TEACHER ###
def get_current_consistency_weight(weight, epoch, rampup):
    """Consistency ramp-up from https://arxiv.org/abs/1610.02242"""
    return weight * sigmoid_rampup(epoch, rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.get_params(), model.get_params()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def cosine_lr(current_epoch, num_epochs, initial_lr):
    return initial_lr * cosine_rampdown(current_epoch, num_epochs)

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def adjust_learning_rate_mean_teacher(optimizer, epoch, num_epochs,
                                      initial_lr_rampup, initial_lr):
    if initial_lr_rampup > 0:
        if epoch <= initial_lr_rampup:
            lr = initial_lr * sigmoid_rampup(epoch, initial_lr_rampup)
        else:
            lr = cosine_lr(epoch-initial_lr_rampup,
                           num_epochs-initial_lr_rampup,
                           initial_lr)
    else:
        lr = cosine_lr(epoch, num_epochs, initial_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def create_model(opt, ema=False):
    model = SCAN(opt, ema)

    return model



if __name__ == '__main__':
    main()
