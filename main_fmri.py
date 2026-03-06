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

from model.proto2mri import Proto2MRI, Proto2MRI_conv
from model.prototype_learning import DinoPro_slim
from util.dataloader import worker_init_fn, NSD_proto, NSD_proto_multiple
from util.proto2mri_trainer import single_epoch_training, evaluate
import cv2 
import torch.nn.functional as F
# import lpips
from glob import glob
from tqdm import tqdm
import pickle
import h5py
import json
from einops import rearrange

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
    parser.add_argument('--topk', default=128, type=int,
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
    parser.add_argument('--selected_proto', default=None, type=str,
                        help='Json file containing the prototypes for ablation')   
    
    parser.add_argument('--decoder_dim', default=768, type=int,
                        help='dimension of encoder features')
    parser.add_argument('--encoder_depth', default=4, type=int,
                    help='number of self-attention layers for visual encoder block')
    parser.add_argument('--decoder_depth', default=1, type=int,
                    help='number of self-attention layers for visual decoder block')
    parser.add_argument('--num_heads', default=8, type=int,
                        help='number of attention heads')
    parser.add_argument('--hemi', default='lh', type=str,
                    help='hemisphere')    
    parser.add_argument('--save_attention', default=None,
                        help='file for saving the attention maps')
    return parser.parse_args()

def main(args):
    # tensorboard initialization
    log_writer = SummaryWriter(log_dir=args.log_dir)

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    # train_dataset = NSD_proto(args.proto_dir, args.mri_dir, args.subject, args.area_name, args.patch_name, seed=args.seed, split='train')
    # val_dataset = NSD_proto(args.proto_dir, args.mri_dir, args.subject, args.area_name, args.patch_name, seed=args.seed, split='val')

    # for training multi-area model for functional clustering
    train_dataset = NSD_proto_multiple(args.proto_dir, args.mri_dir, args.subject, ['faces', 'bodies', 'places'], 
                                       ['OFA', 'EBA', 'OPA'], seed=args.seed, split='train', hemi=args.hemi)
    val_dataset = NSD_proto_multiple(args.proto_dir, args.mri_dir, args.subject, ['faces', 'bodies', 'places'], 
                                        ['OFA', 'EBA', 'OPA'], seed=args.seed, split='val', hemi=args.hemi)


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

    # model = Proto2MRI(num_prototype=args.num_prototype, num_voxels=train_dataset.num_voxels)
    model = Proto2MRI_conv(num_prototype=args.num_prototype, num_voxels=train_dataset.num_voxels)

    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # optimization parameters
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_without_ddp.parameters()), 
                                lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)    
    loss_scaler = NativeScaler()

    iteration = 0
    best_score = -float('inf')

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
        
        if cur_score > best_score:
            best_score = cur_score
            torch.save({
                    'epoch': epoch,
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }, os.path.join(args.log_dir, 'best_model.pt'))

def proto_permutation(args):
    """ Ablating diverse proto responses based on their connection to certain brain area
    """
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    all_dataset = NSD_proto(args.proto_dir, args.mri_dir, args.subject, args.area_name, 
                args.patch_name, seed=args.seed, split='all')

    data_loader = torch.utils.data.DataLoader(
        all_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=2,
        shuffle=True
    )

    model = DinoPro_slim(decoder_dim=args.decoder_dim, encoder_depth=args.encoder_depth, 
                    decoder_depth=args.decoder_depth, num_head=args.num_heads, 
                    num_prototypes=args.num_prototype, topk=args.topk)

    model.load_state_dict(torch.load(args.weights, map_location='cpu')['model'], strict=True)
    proto_vector = model.prototypes.weight.data

    # load selected proto for ablation
    selected_proto = json.load(open(args.selected_proto))
    selected_proto = torch.tensor(selected_proto, dtype=torch.long)

    all_features = []
    all_selected_activation = []
    all_img_names = []
    os.makedirs(args.save_dir, exist_ok=True)

    # --------------------------------------------------
    # 1. Collect all features and names
    # --------------------------------------------------
    for batch_idx, (proto_data, mri_data, img_name) in enumerate(data_loader):
        all_features.append(proto_data)
        all_img_names.extend(img_name)  # store names

        selected_act = proto_data[:, :, selected_proto].mean([1, 2])
        all_selected_activation.append(selected_act.detach().cpu())

    all_features = torch.cat(all_features, dim=0)              
    all_selected_activation = torch.cat(all_selected_activation, dim=0)
    proto_vector = proto_vector.cpu()

    # --------------------------------------------------
    # 2. Select top-200 highest activation samples
    # --------------------------------------------------
    topk_vals, topk_indices = torch.topk(all_selected_activation, 200)

    # --------------------------------------------------
    # 3. Select 200 random zero-activation samples
    # --------------------------------------------------
    zero_mask = (all_features[:, :, selected_proto].sum(dim=-1) == 0)
    zero_indices = torch.where(zero_mask)[0]

    if len(zero_indices) < 200:
        raise ValueError("Not enough zero-activation samples found.")

    random_zero_indices = zero_indices[torch.randperm(len(zero_indices))[:200]]

    # --------------------------------------------------
    # 4. First Set (Removal)
    # --------------------------------------------------
    top_original = all_features[topk_indices].clone()
    top_img_names = [all_img_names[i] for i in topk_indices.tolist()]

    top_permuted = top_original.clone()
    top_permuted[:, :, selected_proto] = 0.0

    top_sum = top_permuted.sum(dim=-1, keepdim=True)
    top_permuted = top_permuted / (top_sum + 1e-8)

    b, n, c = top_original.shape
    top_original = rearrange(top_original, 'b n c -> (b n) c')
    top_permuted = rearrange(top_permuted, 'b n c -> (b n) c')

    top_original_proj = top_original @ proto_vector
    top_permuted_proj = top_permuted @ proto_vector

    top_original_proj = rearrange(top_original_proj, '(b n) c -> b n c', b=b, n=n)
    top_permuted_proj = rearrange(top_permuted_proj, '(b n) c -> b n c', b=b, n=n)

    removal_dict = {
        "original": top_original_proj,
        "permuted": top_permuted_proj,
        "img_name": top_img_names
    }

    with open(os.path.join(args.save_dir, 'removal.pkl'), 'wb') as f:
        pickle.dump(removal_dict, f)

    # --------------------------------------------------
    # 5. Second Set (Addition)
    # --------------------------------------------------
    add_original = all_features[random_zero_indices].clone()
    add_img_names = [all_img_names[i] for i in random_zero_indices.tolist()]

    add_permuted = add_original.clone()

    num_selected = len(selected_proto)
    addition_value = 2 / num_selected
    add_permuted[:, :, selected_proto] = addition_value

    add_sum = add_permuted.sum(dim=-1, keepdim=True)
    add_permuted = add_permuted / (add_sum + 1e-8)

    b, n, c = add_original.shape
    add_original = rearrange(add_original, 'b n c -> (b n) c')
    add_permuted = rearrange(add_permuted, 'b n c -> (b n) c')

    add_original_proj = add_original @ proto_vector
    add_permuted_proj = add_permuted @ proto_vector

    add_original_proj = rearrange(add_original_proj, '(b n) c -> b n c', b=b, n=n)
    add_permuted_proj = rearrange(add_permuted_proj, '(b n) c -> b n c', b=b, n=n)

    addition_dict = {
        "original": add_original_proj,
        "permuted": add_permuted_proj,
        "img_name": add_img_names
    }

    with open(os.path.join(args.save_dir, 'addition.pkl'), 'wb') as f:
        pickle.dump(addition_dict, f)

    print("Saved removal.pkl and addition.pkl successfully.")


def get_spatial_attention(args):

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    dataset = NSD_proto(args.proto_dir, args.mri_dir, args.subject, args.area_name, args.patch_name, seed=args.seed, split='val')

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=36,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False, 
        worker_init_fn=worker_init_fn,
        prefetch_factor=2,
        shuffle=False
    )  

    # model = Proto2MRI(num_prototype=args.num_prototype, num_voxels=dataset.num_voxels)
    model = Proto2MRI_conv(num_prototype=args.num_prototype, num_voxels=dataset.num_voxels)

    model_weight = torch.load(args.weights)['model']
    model.load_state_dict(model_weight)
    model.to(device)
    model.eval()

    topk = 10
    proto_weight = model_weight['proto_vector.weight'].cpu().numpy()
    top_proto = np.argsort(proto_weight.mean(0))[::-1][:topk].tolist()

    # collecting spatial attention maps
    attention_map = dict()
    with torch.no_grad():
        for batch_idx, (proto_data, mri_data, img_name) in enumerate(data_loader):

            # # use model spatial attention
            # proto_data = proto_data.to(device, non_blocking=True).float()
            # att = model.get_att(proto_data)
            # att = att.reshape(len(att), 8, 8) # assuming slim version
            # att = att.data.cpu().numpy()

            # use proto activation
            selected_proto = proto_data[:, :, top_proto]
            att = selected_proto.mean(-1)
            att /= att.sum(-1, keepdim=True)
            att = att.reshape(len(proto_data), 8, 8).numpy()

            for i in range(len(att)):
                attention_map[img_name[i]] = att[i]

    # save the attention maps into pickle

    with open(args.save_attention, 'wb') as f:
        pickle.dump(attention_map, f)



if __name__ == '__main__':  
    args = get_args_parser()
    if args.mode == 'train':
        main(args)
    elif args.mode == 'proto_ablation':
        proto_permutation(args)
    elif args.mode == 'get_att':
        get_spatial_attention(args)