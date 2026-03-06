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

def single_epoch_training(iteration: int, model: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (visual_data, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    # for data_iter_step, (visual_data, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        visual_data = visual_data.to(device, non_blocking=True).float()        
        visual_pred = model(visual_data)

        # # loss = F.mse_loss(visual_pred[:, :-1], visual_data[:, 1:])
        # loss = F.mse_loss(visual_pred, visual_data)

        # for downsampled latent
        patched_visual = rearrange(visual_data, 'b (h w) d -> b h w d', h = 16, w=16)
        patched_visual = rearrange(patched_visual, 'b (h p1) (w p2) d -> b (h w) (d p1 p2)', h = 8, w=8)
        loss = F.mse_loss(visual_pred[:, :-1], patched_visual[:, 1:]) # next patch prediction

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        loss_scaler(loss, optimizer, clip_grad=args.clip,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)


        if log_writer is not None and (data_iter_step + 1) % 10 == 0:
            log_writer.add_scalar('loss:', loss_value_reduce, iteration)
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
    record_visual = 0
    total = 0

    for batch_idx, (visual_data, _) in enumerate(data_loader):
    # for batch_idx, (visual_data, _, _) in enumerate(data_loader):

        visual_data = visual_data.to(device, non_blocking=True).float()
        visual_pred = model(visual_data)

        # visual_mse = F.mse_loss(visual_pred[:, :-1], visual_data[:, 1:])
        # visual_mse = F.mse_loss(visual_pred, visual_data)

        patched_visual = rearrange(visual_data, 'b (h w) d -> b h w d', h = 16, w=16)
        patched_visual = rearrange(patched_visual, 'b (h p1) (w p2) d -> b (h w) (d p1 p2)', h = 8, w=8)
        visual_mse = F.mse_loss(visual_pred[:, :-1], patched_visual[:, 1:])

        record_visual += visual_mse.data.cpu().numpy()*len(visual_data)
        total += len(visual_data)

    record_visual /= total

    log_writer.add_scalar('Validation MSE', record_visual.item(), iteration)

    print("Validation visual MSE:", record_visual.item())
    return record_visual.item()