# coding=utf-8

import os
import sys
import time
import argparse
import random
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# Path Configuration
# ==============================================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_registry import get_baseline_models
from dataset.pde_dataset_ShallowWater import ShallowWaterSolver
from losses import UnweightedL1Loss, UnweightedSquaredL2Loss, UnweightedL2Loss

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==============================================================================
# PDE Dataset
# ==============================================================================
class PdeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dt,
        nsteps,
        dims=(384, 768),
        grid="equiangular",
        pde="shallow water equations",
        initial_condition="random",
        num_examples=32,
        device=torch.device("cpu"),
        normalize=True,
        stream=None,
        provided_stats=None, 
    ):
        self.num_examples = num_examples
        self.device = device
        self.stream = stream

        self.nlat = dims[0]
        self.nlon = dims[1]
        self.grid = grid

        self.nsteps = nsteps
        self.normalize = normalize
        self.nsteps_multiplier = 1 
        
        if pde == "shallow water equations":
            from math import ceil
            lmax = ceil(self.nlat / 3)
            mmax = lmax
            dt_solver = dt / float(self.nsteps)
            
            self.solver = ShallowWaterSolver(self.nlat, self.nlon, dt_solver, lmax=lmax, mmax=mmax, grid=grid).to(self.device).float()
        else:
            raise NotImplementedError

        self.set_initial_condition(ictype=initial_condition)

        # ---------------------------------------------------------
        # Core Normalization Logic
        # ---------------------------------------------------------
        if self.normalize:
            if provided_stats is not None:
                self.inp_mean = provided_stats['inp_mean'].to(self.device)
                self.inp_var = provided_stats['inp_var'].to(self.device)
            else:
                num_estimate = min(100, self.num_examples)
                temp_inps = []
                for _ in range(num_estimate):
                    inp, _ = self._get_sample()
                    temp_inps.append(inp)
                
                temp_inps = torch.stack(temp_inps) 
                
                self.inp_mean = torch.mean(temp_inps, dim=(0, -1, -2)).reshape(-1, 1, 1)
                self.inp_var = torch.var(temp_inps, dim=(0, -1, -2)).reshape(-1, 1, 1)
                
                del temp_inps
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def __len__(self):
        length = self.num_examples if self.ictype == "random" else 1
        return length

    def set_initial_condition(self, ictype="random"):
        self.ictype = ictype

    def set_num_examples(self, num_examples=32):
        self.num_examples = num_examples

    def _get_sample(self):
        if self.ictype == "random":
            inp = self.solver.random_initial_condition(mach=0.2)
        elif self.ictype == "galewsky":
            inp = self.solver.galewsky_initial_condition()

        tar = self.solver.timestep(inp, self.nsteps * self.nsteps_multiplier)
        
        inp = self.solver.gethuv(inp)
        tar = self.solver.gethuv(tar)

        return inp, tar

    def __getitem__(self, index):
        with torch.inference_mode():
            with torch.no_grad():
                inp, tar = self._get_sample()

                if self.normalize:
                    inp = (inp - self.inp_mean) / torch.sqrt(self.inp_var)
                    tar = (tar - self.inp_mean) / torch.sqrt(self.inp_var)

        return inp.clone(), tar.clone()

# ==============================================================================
# Core Training Function 
# ==============================================================================
def train_model(model, train_loader, train_sampler, valid_loader, loss_fn, metrics_fns, optimizer, gscaler, 
                nepochs=20, nfuture=0, amp_mode="none", device=torch.device("cpu"), 
                rank=0, world_size=1, exp_dir=None, 
                start_epoch=0, best_valid_loss=float('inf'), stage_name="pretrain"):
    
    amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16 if amp_mode == "bf16" else torch.float32

    if rank == 0 and exp_dir is not None:
        history_csv_path = os.path.join(exp_dir, f"{stage_name}_history.csv")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, nepochs):
        epoch_start_time = time.time()
        if train_sampler: train_sampler.set_epoch(epoch)
        
        # --- Training Phase ---
        model.train()
        train_loss_local = 0.0
        train_count_local = 0
        
        for inp, tar in train_loader:
            inp, tar = inp.to(device), tar.to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_mode != "none")):
                prd = model(inp)
                for _ in range(nfuture): prd = model(prd.clone())
                loss = loss_fn(prd, tar)

            optimizer.zero_grad(set_to_none=True)
            gscaler.scale(loss).backward()
            gscaler.step(optimizer)
            gscaler.update()
            
            train_loss_local += loss.item() * inp.size(0)
            train_count_local += inp.size(0)

        # --- Validation Phase ---
        model.eval()
        stats = torch.zeros(7, device=device)
        stats[5] = train_loss_local
        stats[6] = train_count_local

        with torch.no_grad():
            for inp, tar in valid_loader:
                inp, tar = inp.to(device), tar.to(device)
                prd = model(inp)
                for _ in range(nfuture): prd = model(prd)
                
                bsz = inp.size(0)
                stats[0] += loss_fn(prd, tar).item() * bsz
                stats[1] += metrics_fns["MAE"](prd, tar).item() * bsz
                stats[2] += metrics_fns["MSE"](prd, tar).item() * bsz
                stats[3] += metrics_fns["RMSE"](prd, tar).item() * bsz
                stats[4] += bsz

        if world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        
        global_train_loss = stats[5].item() / (stats[6].item() + 1e-9)
        global_val_loss = stats[0].item() / (stats[4].item() + 1e-9)
        
        global_val_mae = stats[1].item() / (stats[4].item() + 1e-9)
        global_val_mse = stats[2].item() / (stats[4].item() + 1e-9)
        global_val_rmse = stats[3].item() / (stats[4].item() + 1e-9)
        
        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Output Logic & CSV Writing ---
        if rank == 0:
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
            
            checkpoint = {
                "epoch": epoch + 1,  
                "stage": stage_name,
                "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": gscaler.state_dict(),
                "best_valid_loss": best_valid_loss
            }

            best_tag = ""
            if global_val_loss < best_valid_loss:
                best_valid_loss = global_val_loss
                checkpoint["best_valid_loss"] = best_valid_loss 
                best_tag = " << [NEW BEST]"
                if exp_dir:
                    torch.save(checkpoint, os.path.join(exp_dir, f"best_model_{stage_name}.pt"))

            if exp_dir:
                log_data = {
                    "epoch": epoch + 1, 
                    "time": epoch_duration, 
                    "gpu_mem_gb": peak_mem, 
                    "train_loss": global_train_loss, 
                    "valid_loss": global_val_loss, 
                    "lr": current_lr,
                    "MAE": global_val_mae,   
                    "MSE": global_val_mse,   
                    "RMSE": global_val_rmse  
                }
                write_header = not os.path.exists(history_csv_path)
                pd.DataFrame([log_data]).to_csv(history_csv_path, mode='a', header=write_header, index=False)
                
                torch.save(checkpoint, os.path.join(exp_dir, "latest_model.pt"))

            print(f"Epoch {epoch:03d} | Time: {epoch_duration:.2f}s | GPU Mem: {peak_mem:.2f} GB")
            print(f"  Metrics -> Train Loss: {global_train_loss:.6f} | Val Loss: {global_val_loss:.6f} | MAE: {global_val_mae:.6f} | MSE: {global_val_mse:.6f} | RMSE: {global_val_rmse:.6f}{best_tag}")
                
    return best_valid_loss

# ==============================================================================
# Main Function
# ==============================================================================
def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    nlat, nlon = (256, 512) 
    
    # ---------------------------------------------------------
    # Dataset normalization statistics processing
    # ---------------------------------------------------------
    os.makedirs(args.root_path, exist_ok=True)
    stats_path = os.path.join(args.root_path, "dataset_stats.pt")
    provided_stats = None
    
    if os.path.exists(stats_path):
        if rank == 0:
            print(f"Detected historical normalization stats, loading from {stats_path}...")
        provided_stats = torch.load(stats_path, map_location="cpu")
    else:
        if rank == 0:
            print(f"No historical normalization stats detected, preparing to calculate via sampling and save...")

    train_dataset = PdeDataset(dt=3600, nsteps=150, dims=(nlat, nlon), num_examples=1024, device=device, normalize=True, provided_stats=provided_stats)
    valid_dataset = PdeDataset(dt=3600, nsteps=150, dims=(nlat, nlon), num_examples=256, device=device, normalize=True, provided_stats=provided_stats)
    
    valid_dataset.inp_mean = train_dataset.inp_mean
    valid_dataset.inp_var = train_dataset.inp_var
    
    if provided_stats is None and rank == 0:
        torch.save({
            'inp_mean': train_dataset.inp_mean.cpu(),
            'inp_var': train_dataset.inp_var.cpu()
        }, stats_path)
        print(f"Newly calculated normalization stats saved to: {stats_path}")

    loss_fn = UnweightedSquaredL2Loss().to(device)

    metrics_fns = {
        "MAE": UnweightedL1Loss().to(device),
        "MSE": UnweightedSquaredL2Loss().to(device),
        "RMSE": UnweightedL2Loss().to(device),
    }

    baseline_models = get_baseline_models(img_size=(nlat, nlon), in_chans=3, out_chans=3, res_pred=True)
    
    models_to_run = args.models
    
    for m in models_to_run:
        if m not in baseline_models:
            raise ValueError(f"Model '{m}' not found in baseline_models registry.")
            
    selected_models = {k: baseline_models[k] for k in models_to_run}

    for model_name, model_handle in selected_models.items():
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        model = model_handle().to(device)
        if rank == 0: print(f"\nModel: {model_name} | Params: {count_parameters(model)}")
        
        exp_dir = os.path.join(args.root_path, model_name)
        if rank == 0: os.makedirs(exp_dir, exist_ok=True)

        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)
        gscaler = torch.cuda.amp.GradScaler(enabled=(args.amp_mode != "none"))

        start_epoch = 0
        best_valid_loss = float('inf')
        current_stage = "pretrain"

        # ---------------------------------------------------------
        # Load checkpoint
        # ---------------------------------------------------------
        if args.resume: 
            resume_path = os.path.join(exp_dir, "latest_model.pt")
            if os.path.exists(resume_path): 
                ckpt = torch.load(resume_path, map_location=device)
                
                model.module.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                gscaler.load_state_dict(ckpt['scaler_state_dict'])
                start_epoch = ckpt.get('epoch', 0)
                best_valid_loss = ckpt.get('best_valid_loss', float('inf'))
                current_stage = ckpt.get('stage', 'pretrain')
                
                if rank == 0: 
                    print(f"Resuming from checkpoint {resume_path} (Stage: {current_stage}, Epoch: {start_epoch}, Best Loss: {best_valid_loss:.6f})")

        # ==========================================
        # 1-step pretrain
        # ==========================================
        if args.pretrain_epochs > 0 and current_stage == "pretrain":
            if rank == 0: print(f"\n--- Starting Stage 1: Pretraining (1-step) [Epochs {start_epoch} to {args.pretrain_epochs}] ---")
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.pretrain_lr
                
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler)
            
            best_valid_loss = train_model(model, train_loader, train_sampler, valid_loader, loss_fn, metrics_fns, optimizer, gscaler, 
                        nepochs=args.pretrain_epochs, nfuture=0, amp_mode=args.amp_mode, device=device, rank=rank, world_size=world_size, 
                        exp_dir=exp_dir, start_epoch=start_epoch, best_valid_loss=best_valid_loss, stage_name="pretrain")
            
            start_epoch = 0
            best_valid_loss = float('inf')
            current_stage = "finetune"

        # ==========================================
        # 2-step finetune (Autoregressive)
        # ==========================================
        if args.finetune_epochs > 0 and current_stage == "finetune":
            if rank == 0: print(f"\n--- Starting Stage 2: Finetuning (2-step Autoregressive) [Epochs {start_epoch} to {args.finetune_epochs}] ---")
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.finetune_lr
                
            train_dataset.nsteps_multiplier = 2
            valid_dataset.nsteps_multiplier = 2
            
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler)
            
            train_model(model, train_loader, train_sampler, valid_loader, loss_fn, metrics_fns, optimizer, gscaler, 
                        nepochs=args.finetune_epochs, nfuture=1, amp_mode=args.amp_mode, device=device, rank=rank, world_size=world_size, 
                        exp_dir=exp_dir, start_epoch=start_epoch, best_valid_loss=best_valid_loss, stage_name="finetune")

    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--models", type=str, nargs='+', required=True, help="List of models to run")
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pretrain_lr", type=float, default=2e-3, help="Learning rate for stage 1 (1-step pretrain)")
    parser.add_argument("--finetune_lr", type=float, default=5e-4, help="Learning rate for stage 2 (autoregressive finetune)")
    parser.add_argument("--amp_mode", type=str, default="none")
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())