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

from model.prototype_learning import DinoPro, DinoPro_slim
from util.dataloader import COCO_feat, worker_init_fn, NSD_feat, ImageNet_Feature
from util.prototype_trainer import single_epoch_training, evaluate
from util.clipscore_clustering import clipscore_clustering
import cv2 
import torch.nn.functional as F
# import lpips
from glob import glob
from tqdm import tqdm
import pickle
import h5py


def get_args_parser():
    parser = argparse.ArgumentParser('Master script for prototype learning with DINO v3 features',
                                     add_help=False)
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
    parser.add_argument('--feat_path', default=None, type=str,
                        help='path to dino features')
    parser.add_argument('--image_path', default=None, type=str,
                        help='path to coco images for prototype visualization')
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
    parser.add_argument('--decoder_dim', default=768, type=int,
                        help='dimension of encoder features')
    parser.add_argument('--encoder_depth', default=4, type=int,
                    help='number of self-attention layers for visual encoder block')
    parser.add_argument('--decoder_depth', default=1, type=int,
                    help='number of self-attention layers for visual decoder block')
    parser.add_argument('--num_heads', default=8, type=int,
                        help='number of attention heads')
    parser.add_argument('--clip', default=1, type=float,
                        help='clip grad')
    parser.add_argument('--top_image', default=20, type=int,
                        help='top images to represent each prototype')
    parser.add_argument('--candidate_image', default=50, type=int,
                        help='number of candidate images to select top-k from for each prototype')
    return parser.parse_args()

def main(args):
    # tensorboard initialization
    log_writer = SummaryWriter(log_dir=args.log_dir)

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    train_dataset = COCO_feat(args.feat_path, split='train')
    val_dataset = COCO_feat(args.feat_path, split='val')


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

    # model = DinoPro(decoder_dim=args.decoder_dim, encoder_depth=args.encoder_depth, 
    #                 decoder_depth=args.decoder_depth, num_head=args.num_heads, 
    #                 num_prototypes=args.num_prototype, topk=args.topk)

    model = DinoPro_slim(decoder_dim=args.decoder_dim, encoder_depth=args.encoder_depth, 
                    decoder_depth=args.decoder_depth, num_head=args.num_heads, 
                    num_prototypes=args.num_prototype, topk=args.topk)
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
        
        if (epoch+1)%10 == 0:
            cur_score = evaluate(epoch, iteration, model, 
                                data_loader_val, device, 
                                log_writer, args)
            
            if cur_score < best_score:
                best_score = cur_score
                torch.save({
                        'epoch': epoch,
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        }, os.path.join(args.log_dir, 'best_model.pt'))

            
        torch.save({
                    'epoch': epoch,
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }, os.path.join(args.log_dir, 'model.pt'))
        


def get_top_image(args):
    """ Get the top-20 images with highest responses per prototype, 
     and save the results as a pickle for downstream visualization purposes.
    """

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    test_dataset = COCO_feat(args.feat_path, split='val')
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset, 
        # sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True, 
        worker_init_fn=worker_init_fn,
        prefetch_factor=2,
        shuffle=False
    )  

    # model = DinoPro(decoder_dim=args.decoder_dim, encoder_depth=args.encoder_depth, 
    #                 decoder_depth=args.decoder_depth, num_head=args.num_heads, 
    #                 num_prototypes=args.num_prototype, topk=args.topk)

    model = DinoPro_slim(decoder_dim=args.decoder_dim, encoder_depth=args.encoder_depth, 
                    decoder_depth=args.decoder_depth, num_head=args.num_heads, 
                    num_prototypes=args.num_prototype, topk=args.topk)

    model.load_state_dict(torch.load(args.weights, map_location='cpu')['model'], strict=True)
    model.to(device)
    model.eval()

    with torch.no_grad():
        record_feat = []
        record_image_name = []
        for batch_idx, (visual_data, img_name) in enumerate(tqdm(data_loader_test)):
            visual_data = visual_data.to(device, non_blocking=True).float()
            proto_sim = model.get_prototype_similarity(visual_data) 
            proto_sim = proto_sim.cpu().numpy() # (b, n, num_prototypes)
            record_feat.extend(proto_sim)
            record_image_name.extend(img_name)
        
    record_feat = np.array(record_feat) # (N, n, num_prototypes)
    record_image_name = np.array(record_image_name) # (N,)

    # find the top-k images for each prototype (use simple average for now)
    avg_response = record_feat.mean(axis=1) # (N, num_prototypes)
    topk_idx = np.argsort(avg_response, axis=0)[::-1][:args.candidate_image, :] # (topk, num_prototypes)
    topk_image_name = record_image_name[topk_idx] # (topk, num_prototypes)

    # run clip score clustering
    topk_image_name = clipscore_clustering(topk_image_name, args.top_image, args.image_path)

    # # max instead of average for finding the top-k images
    # topk = top_image  # number of images to select per prototype
    # max_response = record_feat.max(axis=1)   # (M, num_prototypes)
    # top5_proto_idx = np.argsort(max_response, axis=1)[:, -5:]  # (M, 5)
    # M, N = max_response.shape
    # mask = np.zeros((M, N), dtype=bool)
    # rows = np.arange(M)[:, None]
    # mask[rows, top5_proto_idx] = True
    # filtered_response = np.where(mask, max_response, -np.inf)  # (M, num_prototypes)
    # topk_idx = np.argsort(filtered_response, axis=0)[::-1][:topk, :]  # (topk, num_prototypes)
    # topk_image_name = record_image_name[topk_idx]  # (topk, num_prototypes)

    # # reorganize the prototype based on its average activation
    # overall_avg_response = record_feat.mean(axis=(0, 1)) # (num_prototypes,)
    # proto_order = np.argsort(overall_avg_response)[::-1] # (num_prototypes,)

    # now also get the attention maps for the top-k images for each prototypes
    topk_attention_maps = []
    for i in tqdm(range(args.num_prototype)):
        # cur_proto_idx = proto_order[i]
        cur_proto_idx = i
        cur_topk_image_name = topk_image_name[:, cur_proto_idx]
        cur_attention_maps = []
        for img_name in cur_topk_image_name:
            idx = np.where(record_image_name == img_name)[0][0]
            feat = record_feat[idx] # (n, num_prototypes)
            # attention_map = feat[:, cur_proto_idx].reshape(16, 16) # (16, 16)
            attention_map = feat[:, cur_proto_idx].reshape(8, 8) # (16, 16)
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8) # normalize to [0, 1]
            cur_attention_maps.append(attention_map)
        topk_attention_maps.append(cur_attention_maps)

    topk_attention_maps = np.array(topk_attention_maps) # (num_prototypes, topk, 16, 16)

    save_pkl = {'image_name': topk_image_name.T,
                'attention_map': topk_attention_maps
                }

    with open(os.path.join(args.log_dir, 'topk_images_clipscore.pkl'), 'wb') as f:
        pickle.dump(save_pkl, f)


def get_nsd_proto(args):
    """ Get the prototype similarity for NSD data for prototype -> fMRI training.
    """
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    test_dataset = NSD_feat(args.feat_path)
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset, 
        # sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False, 
        worker_init_fn=worker_init_fn,
        prefetch_factor=2,
        shuffle=False
    )  

    # model = DinoPro(decoder_dim=args.decoder_dim, encoder_depth=args.encoder_depth, 
    #                 decoder_depth=args.decoder_depth, num_head=args.num_heads, 
    #                 num_prototypes=args.num_prototype, topk=args.topk)
    model = DinoPro_slim(decoder_dim=args.decoder_dim, encoder_depth=args.encoder_depth, 
                    decoder_depth=args.decoder_depth, num_head=args.num_heads, 
                    num_prototypes=args.num_prototype, topk=args.topk)
    
    model.load_state_dict(torch.load(args.weights, map_location='cpu')['model'], strict=True)
    model.to(device)
    model.eval()

    record_feat = {subj_name: [] for subj_name in np.unique(test_dataset.all_subj)}
    record_image_name = {subj_name: [] for subj_name in np.unique(test_dataset.all_subj)}
    with torch.no_grad():
        for batch_idx, (visual_data, img_name, subj_name) in enumerate(tqdm(data_loader_test)):
            visual_data = visual_data.to(device, non_blocking=True).float()
            proto_sim = model.get_prototype_similarity(visual_data) 
            proto_sim = proto_sim.cpu().numpy() # (b, n, num_prototypes)

            for i in range(len(subj_name)):
                record_feat[subj_name[i]].append(proto_sim[i])
                record_image_name[subj_name[i]].append(img_name[i])
    
    os.makedirs(os.path.join(args.log_dir, 'nsd_proto'), exist_ok=True)
    for subj_name in record_feat.keys():
        record_feat[subj_name] = np.array(record_feat[subj_name]) # (N, n, num_prototypes)
        record_image_name[subj_name] = np.array(record_image_name[subj_name]) # (N,)

        save_pkl = {'image_name': record_image_name[subj_name],
                    'prototype_similarity': record_feat[subj_name]
                    }
        with open(os.path.join(args.log_dir, 'nsd_proto', subj_name+'.pkl'), 'wb') as f:
            pickle.dump(save_pkl, f)

def get_nsd_latent(args):
    """ Get latent features for NSD data, for training the pixel readout models.
    """
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    # all_dataset = NSD_feat(args.feat_path)

    all_dataset = ImageNet_Feature(args.feat_path,
                    'all', sampled_img_num=500) # sample 500 images per category for faster training
    

    data_loader = torch.utils.data.DataLoader(
        all_dataset, 
        # sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False, 
        worker_init_fn=worker_init_fn,
        prefetch_factor=2,
        shuffle=False
    )  

    # model = DinoPro(decoder_dim=args.decoder_dim, encoder_depth=args.encoder_depth, 
    #                 decoder_depth=args.decoder_depth, num_head=args.num_heads, 
    #                 num_prototypes=args.num_prototype, topk=args.topk)

    model = DinoPro_slim(decoder_dim=args.decoder_dim, encoder_depth=args.encoder_depth, 
                    decoder_depth=args.decoder_depth, num_head=args.num_heads, 
                    num_prototypes=args.num_prototype, topk=args.topk)

    model.load_state_dict(torch.load(args.weights, map_location='cpu')['model'], strict=True)
    model.to(device)
    model.eval()

    num_samples = len(data_loader.dataset)
    h5_path = args.save_dir

    # Ensure directory exists
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    with h5py.File(h5_path, "w") as h5f:

        # Create feature dataset: 80k × 256 × 768
        dset_features = h5f.create_dataset(
            "features",
            shape=(num_samples, 64, 768),
            dtype="float32",
            chunks=(1, 64, 768),
            compression="lzf"
        )

        # Create string datasets
        str_dt = h5py.string_dtype(encoding="utf-8")
        dset_names = h5f.create_dataset("names", shape=(num_samples,), dtype=str_dt)
        dset_category = h5f.create_dataset("subjects", shape=(num_samples,), dtype=str_dt)

        idx = 0

        # IMPORTANT: HDF5 writing happens only in the main process
        # The model inference happens in the main process as well.
        with torch.no_grad():
            for batch_idx, (visual_data, img_name, subject_name) in tqdm(
                    enumerate(data_loader), total=len(data_loader), desc="Processing"):
                B = len(visual_data)

                # Move batch to CUDA
                visual_data = visual_data.to(device, non_blocking=True).float()

                # Run inference in the MAIN PROCESS — safe
                proto_latent = model.get_latents(visual_data)

                # Move back to CPU for writing
                proto_latent = proto_latent.cpu().numpy()   # shape (B, 256, 768)

                # Write into HDF5 (main process only)
                dset_features[idx:idx + B, :, :] = proto_latent
                dset_names[idx:idx + B] = img_name
                dset_category[idx:idx + B] = subject_name

                idx += B

if __name__ == '__main__':  
    args = get_args_parser()
    if args.mode == 'train':
        main(args)
    elif args.mode == 'get_top_image':
        get_top_image(args)
    elif args.mode == 'get_nsd_proto':
        get_nsd_proto(args)
    elif args.mode == 'get_nsd_latent':
        get_nsd_latent(args)