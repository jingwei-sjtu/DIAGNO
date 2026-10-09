# coding=utf-8

import os
import sys
import time
import argparse
import random
import shutil
import csv
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import numpy as np

# ==============================================================================
# Path Configuration
# ==============================================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_registry import get_baseline_models
from dataset.pde_dataset_ShallowWater_steps import PdeDataset
from losses import (
    UnweightedL1Loss, UnweightedSquaredL2Loss, UnweightedL2Loss
)

# ==============================================================================
# DDP Helper Functions
# ==============================================================================
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# ==============================================================================
# Dataset Wrapper
# ==============================================================================
class SWInferenceWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        inp, tar = self.base_dataset[index]
        return index, inp, tar

# ==============================================================================
# Core Inference Function
# ==============================================================================
def infer_model(model, valid_loader, metrics_fns, device=torch.device("cpu"), 
                rank=0, world_size=1, infer_steps=1, save_steps=None, save_steps_dir=None):
    
    model.eval()
    base_dataset = valid_loader.dataset.base_dataset
    mean = base_dataset.inp_mean.view(1, 3, 1, 1).to(device)
    std = torch.sqrt(base_dataset.inp_var).view(1, 3, 1, 1).to(device)

    stats_metrics = torch.zeros((infer_steps, 3), device=device)
    stats_count = torch.zeros(1, device=device)
    infer_start_time = time.time()

    if rank == 0:
        print(f"\n--- Starting {infer_steps}-step autoregressive inference validation ---")

    if save_steps is not None and save_steps_dir is not None:
        if rank == 0:
            for s in save_steps:
                os.makedirs(os.path.join(save_steps_dir, f"step_{s}", "temp"), exist_ok=True)
        if world_size > 1:
            dist.barrier()

    with torch.no_grad():
        for batch_idx, (global_idx, inp, tar) in enumerate(valid_loader):
            inp, tar = inp.to(device), tar.to(device)
            bsz = inp.size(0)
            stats_count += bsz
            prd = inp
            
            for step in range(infer_steps):
                prd = model(prd)
                current_tar = tar[:, step] if tar.ndim > inp.ndim else tar
                
                prd_denorm = prd * std + mean
                tar_denorm = current_tar * std + mean
                
                if save_steps is not None and (step + 1) in save_steps:
                    temp_dir = os.path.join(save_steps_dir, f"step_{step+1}", "temp")
                    np.save(os.path.join(temp_dir, f"prd_rank{rank}_batch{batch_idx}.npy"), prd_denorm.cpu().numpy())
                    np.save(os.path.join(temp_dir, f"tar_rank{rank}_batch{batch_idx}.npy"), tar_denorm.cpu().numpy())
                    np.save(os.path.join(temp_dir, f"idx_rank{rank}_batch{batch_idx}.npy"), global_idx.numpy())

                stats_metrics[step, 0] += metrics_fns["MAE"](prd_denorm, tar_denorm).item() * bsz
                stats_metrics[step, 1] += metrics_fns["MSE"](prd_denorm, tar_denorm).item() * bsz
                stats_metrics[step, 2] += metrics_fns["RMSE"](prd_denorm, tar_denorm).item() * bsz

    if world_size > 1:
        dist.barrier()
        dist.all_reduce(stats_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(stats_metrics, op=dist.ReduceOp.SUM)

    if rank == 0 and save_steps is not None:
        print("Merging distributed prediction results in absolute temporal order...")
        total_valid_samples = len(valid_loader.dataset)
        
        for s in save_steps:
            step_dir = os.path.join(save_steps_dir, f"step_{s}")
            temp_dir = os.path.join(step_dir, "temp")
            
            idx_files = [f for f in os.listdir(temp_dir) if f.startswith("idx_")]
            final_prd = None
            final_tar = None
            
            for idx_file in idx_files:
                parts = idx_file.replace(".npy", "").split("_")
                r_str, b_str = parts[1], parts[2]
                
                indices = np.load(os.path.join(temp_dir, idx_file))
                prd_batch = np.load(os.path.join(temp_dir, f"prd_{r_str}_{b_str}.npy"))
                tar_batch = np.load(os.path.join(temp_dir, f"tar_{r_str}_{b_str}.npy"))
                
                if final_prd is None:
                    _, c, h, w = prd_batch.shape
                    final_prd = np.zeros((total_valid_samples, c, h, w), dtype=np.float32)
                    final_tar = np.zeros((total_valid_samples, c, h, w), dtype=np.float32)
                
                for i, g_idx in enumerate(indices):
                    if g_idx < total_valid_samples:
                        final_prd[g_idx] = prd_batch[i]
                        final_tar[g_idx] = tar_batch[i]
                        
            np.save(os.path.join(step_dir, f"merged_prd_step{s}.npy"), final_prd)
            np.save(os.path.join(step_dir, f"merged_tar_step{s}.npy"), final_tar)
            shutil.rmtree(temp_dir)
            
        print(f"Merge complete! Saved to: {save_steps_dir}")
    
    total_samples = stats_count.item()

    global_mae = stats_metrics[:, 0] / total_samples
    global_mse = stats_metrics[:, 1] / total_samples
    global_rmse = stats_metrics[:, 2] / total_samples
    
    infer_duration = time.time() - infer_start_time

    if rank == 0:
        print(f"\nInference time: {infer_duration:.2f}s | Total evaluated sequences: {int(total_samples)}")
        print("="*105)
        print(f"{'Step':<5} | {'MAE (Phys)':<12} | {'MSE (Phys)':<12} | {'RMSE (Phys)':<12}")
        print("-" * 105)
        for step in range(infer_steps):
            print(f"{step+1:<5d} | {global_mae[step]:.6f}     | {global_mse[step]:.6f}     | {global_rmse[step]:.6f}")
        print("="*105)
            
    return global_mae.cpu().numpy(), global_mse.cpu().numpy(), global_rmse.cpu().numpy()

# ==============================================================================
# Main Function
# ==============================================================================
def main(args):
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    nlat, nlon = (256, 512) 
    grid = "equiangular"
    
    train_dataset = PdeDataset(dt=3600, nsteps=150, dims=(nlat, nlon), num_examples=2, device=device, normalize=True)
    base_valid_dataset = PdeDataset(dt=3600, nsteps=150, dims=(nlat, nlon), num_examples=args.num_eval_samples, device=device, normalize=True)
    
    stats_path = os.path.join(args.root_path, "dataset_stats.pt")
    if os.path.exists(stats_path):
        if rank == 0:
            print(f"Historical normalization parameters detected, loading from {stats_path}...")
        stats = torch.load(stats_path, map_location=device)
        train_dataset.inp_mean = stats['inp_mean'].to(device)
        train_dataset.inp_var = stats['inp_var'].to(device)
    else:
        if rank == 0:
            print(f"Warning: {stats_path} not detected. Random normalization parameters will be used!")

    base_valid_dataset.inp_mean = train_dataset.inp_mean
    base_valid_dataset.inp_var = train_dataset.inp_var
    base_valid_dataset.nsteps_multiplier = args.infer_steps

    valid_dataset = SWInferenceWrapper(base_valid_dataset)

    metrics_fns = {
        "MAE": UnweightedL1Loss().to(device),
        "MSE": UnweightedSquaredL2Loss().to(device),
        "RMSE": UnweightedL2Loss().to(device),
    }

    baseline_models = get_baseline_models(img_size=(nlat, nlon), in_chans=3, out_chans=3, res_pred=True)
    
    if args.model_name not in baseline_models:
        if rank == 0: print(f"Error: Model {args.model_name} not found in registry.")
        cleanup_ddp()
        return

    model_handle = baseline_models[args.model_name]
    model = model_handle().to(device)
    
    exp_dir = os.path.join(args.root_path, args.model_name)
    weight_path = args.pretrained_path if args.pretrained_path else os.path.join(exp_dir, "best_model.pt")
    
    if os.path.exists(weight_path):
        if rank == 0: print(f"Loading weights from: {weight_path}")
        state_dict = torch.load(weight_path, map_location=device)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        if rank == 0: print(f"Warning: No weights found at {weight_path}. Running with uninitialized weights.")

    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        sampler=valid_sampler
    )
    
    save_steps_dir = os.path.join(exp_dir, f"specific_steps_results_{args.infer_steps}steps") if args.save_steps else None

    mae, mse, rmse = infer_model(model, valid_loader, metrics_fns, device=device, rank=rank, world_size=world_size, 
                infer_steps=args.infer_steps, save_steps=args.save_steps, save_steps_dir=save_steps_dir)

    if rank == 0:
        csv_name = f"metrics_{args.model_name}_{args.infer_steps}steps.csv"
        csv_path = os.path.join(exp_dir, csv_name)
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "MAE_Phys", "MSE_Phys", "RMSE_Phys"])
            for i in range(args.infer_steps):
                writer.writerow([i + 1, mae[i], mse[i], rmse[i]])
        print(f"Inference metrics successfully saved to: {csv_path}")

    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--infer_steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_eval_samples", type=int, default=256)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--save_steps", type=int, nargs='+', default=None, help="Specifies prediction steps where physical quantities should be separately saved")
    main(parser.parse_args())
