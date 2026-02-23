import sys
sys.path.append('./model')
sys.path.append('./util')
sys.path.append('/home/shichen/project/dinov3/')

import argparse
from random import shuffle
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import misc
from misc import NativeScalerWithGradNormCount as NativeScaler

from model.proto2mri import Proto2MRI
from util.dataloader import worker_init_fn, NSD_proto
from util.proto2mri_trainer import single_epoch_training, evaluate
import cv2 
import torch.nn.functional as F
# import lpips
from glob import glob
from tqdm import tqdm
import pickle
import h5py


def get_args_parser():
    parser = argparse.ArgumentParser('Master script for learning the mapping from prototype space to MRI space', add_help=False)
    parser.add_argument('--mode', default='train', type=str,
                        help='running mode')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=90, type=int)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--proto_dir', default=None, type=str,
                        help='path to prototype features')
    parser.add_argument('--mri_dir', default=None, type=str,
                        help='path to NSD root directory')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log and model checkpoints')
    parser.add_argument('--save_freq', default=3, type=int,
                        help='save frequency (epoch)')
    parser.add_argument('--save_dir', default=None, type=str,
                        help='path to save visualization results')
    parser.add_argument('--weights', default=None,
                        help='weights for model')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
        
    # data setting
    parser.add_argument('--num_prototype', default=1000, type=int,
                        help='number of prototypes')
    parser.add_argument('--topk', default=32, type=int,
                        help='topk bottleneck for sparsity constraint')
    parser.add_argument('--clip', default=1, type=float,
                        help='clip grad')
    parser.add_argument('--seed', default=88, type=int,
                        help='random seed for train/val split')
    parser.add_argument('--subject', default='subj01', type=str,
                        help='subject name for training and evaluation')
    parser.add_argument('--area_name', default='bodies', type=str,
                        help='functional area name for experiment')    
    parser.add_argument('--patch_name', default='EBA', type=str,
                        help='specific patch name for experiment, e.g. EBA, FFA, PPA, etc.')   
    parser.add_argument('--alpha', default=1, type=float,
                        help='balance factor') 
    return parser.parse_args()

def main(args):
    # tensorboard initialization
    log_writer = SummaryWriter(log_dir=args.log_dir)

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    train_dataset = NSD_proto(args.proto_dir, args.mri_dir, args.subject, args.area_name, args.patch_name, seed=args.seed, split='train')
    val_dataset = NSD_proto(args.proto_dir, args.mri_dir, args.subject, args.area_name, args.patch_name, seed=args.seed, split='val')

    # code for distributed training, borrowed from MAE repo
    if args.distributed:  # args.distributed
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, 
        # sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=2,
        shuffle=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, 
        # sampler=sampler_val,
        batch_size=36,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True, 
        worker_init_fn=worker_init_fn,
        prefetch_factor=2,
        shuffle=False
    )  

    model = Proto2MRI(num_prototype=args.num_prototype, num_voxels=train_dataset.num_voxels)
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # optimization parameters
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_without_ddp.parameters()), 
                                lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)    
    loss_scaler = NativeScaler()

    iteration = 0
    best_score = float('inf')

    for epoch in range(args.epochs):

        iteration = single_epoch_training(iteration, model,
                                        data_loader_train, 
                                        optimizer, device, epoch, 
                                        loss_scaler, log_writer, args,
                                       )
        

        cur_score = evaluate(epoch, iteration, model, 
                            data_loader_val, device, 
                            log_writer, args)
        
        torch.save({
                    'epoch': epoch,
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }, os.path.join(args.log_dir, 'model.pt'))
        
        if cur_score < best_score:
            best_score = cur_score
            torch.save({
                    'epoch': epoch,
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }, os.path.join(args.log_dir, 'best_model.pt'))

if __name__ == '__main__':  
    args = get_args_parser()
    if args.mode == 'train':
        main(args)