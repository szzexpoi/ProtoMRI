import math
import sys
from typing import Iterable, Optional

import torch
import misc as misc
import lr_sched as lr_sched
import cv2
import os
import numpy as np
import torch.nn.functional as F
import pickle
from einops import rearrange

def pearson_corr_loss(pred, target, eps=1e-8):
    # Flatten
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    # Mean center
    pred_mean = pred - pred.mean(dim=1, keepdim=True)
    target_mean = target - target.mean(dim=1, keepdim=True)

    # Compute correlation
    numerator = (pred_mean * target_mean).sum(dim=1)
    denominator = torch.sqrt(
        (pred_mean ** 2).sum(dim=1) * (target_mean ** 2).sum(dim=1) + eps
    )

    corr = numerator / denominator

    # Loss = 1 - correlation (maximize corr → minimize loss)
    return 1 - corr.mean()

def single_epoch_training(iteration: int, model: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mse', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('pearson', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (proto_data, mri_data, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        proto_data = proto_data.to(device, non_blocking=True).float()        
        mri_data = mri_data.to(device, non_blocking=True).float()

        mri_pred = model(proto_data)
        mse_loss = F.mse_loss(mri_pred, mri_data)
        corr_loss = pearson_corr_loss(mri_pred, mri_data)

        loss = mse_loss + args.alpha*corr_loss

        loss_value = loss.item()
        mse_value = mse_loss.item()
        corr_value = corr_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        loss_scaler(loss, optimizer, clip_grad=args.clip,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(mse=mse_value)
        metric_logger.update(pearson=corr_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        mse_value_reduce = misc.all_reduce_mean(mse_value)
        corr_value_reduce = misc.all_reduce_mean(corr_value)


        if log_writer is not None and (data_iter_step + 1) % 10 == 0:
            log_writer.add_scalar('loss:', loss_value_reduce, iteration)
            log_writer.add_scalar('mse:', mse_value_reduce, iteration)
            log_writer.add_scalar('pearson:', corr_value_reduce, iteration)
            log_writer.add_scalar('lr:', max_lr, iteration)
        iteration += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return iteration

@torch.no_grad()
def evaluate(cur_epoch, iteration, model, data_loader, 
               device, log_writer=None, args=None):
    # switch to evaluation mode
    model.eval()
    record_loss = 0
    record_pearson = 0
    total = 0

    for batch_idx, (proto_data, mri_data, _) in enumerate(data_loader):
        proto_data = proto_data.to(device, non_blocking=True).float()
        mri_data = mri_data.to(device, non_blocking=True).float()
        mri_pred = model(proto_data)
        mri_mse = F.mse_loss(mri_pred, mri_data)
        mri_pearson = 1 - pearson_corr_loss(mri_pred, mri_data) # convert back to correlation
        record_loss += mri_mse.data.cpu().numpy()*len(mri_data)
        record_pearson += mri_pearson.data.cpu().numpy()*len(mri_data)
        total += len(mri_data)

    record_loss /= total
    record_pearson /= total
    log_writer.add_scalar('Validation MSE', record_loss.item(), iteration)
    log_writer.add_scalar('Validation Pearson', record_pearson.item(), iteration)
    print("Validation MSE:", record_loss.item())
    print("Validation Pearson:", record_pearson.item())
    return record_pearson.item()