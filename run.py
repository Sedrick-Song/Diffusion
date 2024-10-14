import os
import random
import argparse
import numpy as np
import torch
import torch.distributed
import torch.optim as optim
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import logging
import torch.utils
import wandb
from datetime import datetime
from pathlib import Path
import librosa

from utils.train import train_one_epoch
from utils.dist import setup_dist, cleanup_dist
from utils.checkpoint import save_checkpoint
from utils.logger import setup_logger
from data.chinese import Chinesedataset
from data.chinese import custom_collate_fn
from model.tokenizer import AudioTokenizer
from model.model import AudioDiffusion
from src.diffusers.models.transformer_2d import Transformer2DModel
from encodec.model import EncodecModel

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=1, help="Number of GPUs for DDP training")
    parser.add_argument("--master_port", type=int, default=12354, help="Master port to use for DDP training")
    parser.add_argument("--wandb", type=bool, default=False, help="Whether to use wandb")
    parser.add_argument("--debug", type=bool, default=True, help="Whether to use wandb")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--exp_dir", type=str, default="/apdcephfs/private_sedricksong/exp/encodec_DiT_debug", help="The experiment dir")
    parser.add_argument("--train_file", type=str, default="/apdcephfs/private_sedricksong/data/zh_magic_data/train_file_part", help="The train file path")
    parser.add_argument("--dev_file", type=str, default="/apdcephfs/private_sedricksong/data/zh_magic_data/dev_file", help="The dev file path")
    parser.add_argument("--wandb_dir", type=str, default="/apdcephfs_cq8/private_sedricksong/wandb", help="The wandb dir")
    parser.add_argument("--wandb_name", type=str, default="songzheshu", help="The wandb entity name")
    parser.add_argument("--wandb_project", type=str, default="encodec_dit", help="The wandb project name")
    parser.add_argument("--wandb_exp_name", type=str, default="exp1", help="The wandb exp name")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="The seed for random generators intended for reproducibility")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--bandwidth", type=int, default=6, help="Encodec bandwidth")
    parser.add_argument("--scheduler_name", type=str, default="stabilityai/stable-diffusion-2-1", help="Scheduler identifier.")
    parser.add_argument("--unet_model_name", type=str, default=None, help="UNet model identifier from huggingface.co/models.")
    parser.add_argument("--unet_model_config", type=str, default="/apdcephfs_cq8/private_sedricksong/Encodec_DiT/configs/diffusion_model_config_dit_L_4.json", help="UNet model config json path.")
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. ""More details here: https://arxiv.org/abs/2303.09556.")
    parser.add_argument("--uncondition", action="store_true", default=False, help="10% uncondition for training.")
    args = parser.parse_args()
    return args

def fix_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

def run(rank, world_size, args):
    # set up random seed
    fix_random_seed(args.seed)
    if world_size > 1:
        setup_dist(rank, world_size, args.master_port)

    # set up logger
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    backup_path = f"{args.exp_dir}/log/log-train/{date_time}"

    setup_logger(f"{args.exp_dir}/log/log-train")
    logging.info("Training started")

    # set wandb
    if args.wandb and rank == 0:
        if not os.path.exists(args.wandb_dir):
            os.makedirs(args.wandb_dir, exist_ok=True)
        wandb.login(key="5a228f9b67ca67ec10303c14847ed072af2b09b9")
        wandb.init(dir=args.wandb_dir, entity=args.wandb_name, project=args.wandb_project, name=args.wandb_exp_name)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    # set up model
    logging.info("About to create model")
    # Encodec model
    #encodec_fn = "/apdcephfs_cq10/share_1297902/user/sedricksong/pretrained_models/VoiceCraft/encodec_4cb2048_giga.th"
    #audio_tokenizer = AudioTokenizer(signature=encodec_fn)
    audio_tokenizer = EncodecModel.encodec_model_24khz()
    audio_tokenizer.set_target_bandwidth(args.bandwidth)
    # DiT model
    #dit_config = Transformer2DModel.load_config(args.dit_config)
    #model = Transformer2DModel.from_config(dit_config, subfolder="unet")
    model = AudioDiffusion(
        args.scheduler_name, args.unet_model_name, args.unet_model_config, args.snr_gamma, args.uncondition
    )
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    model.to(device)
    audio_tokenizer.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.98)

    # load dataset
    logging.info("Loading dataset")
    dataset_train = Chinesedataset(args.train_file)
    dataset_dev = Chinesedataset(args.dev_file)

    train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank)
    dev_sampler = DistributedSampler(dataset_dev, num_replicas=world_size, rank=rank)

    # define dataloder
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers = 4,
        pin_memory = True,
        sampler=train_sampler,
        batch_size = args.batch_size,
        collate_fn = custom_collate_fn,
        shuffle = False
    )

    dev_dataloader = torch.utils.data.DataLoader(
        dataset_dev,
        num_workers = 4,
        pin_memory = True,
        sampler=dev_sampler,
        batch_size = args.batch_size,
        collate_fn = custom_collate_fn,
        shuffle = False
    )

    # training process
    for epoch in range(1, args.num_epochs):
        train_one_epoch(
            encodec=audio_tokenizer,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dataloader,
            valid_dl=dev_dataloader,
            epoch=epoch,
            args=args,
            device=device,
            world_size=world_size,
            rank=rank
        )

        # save checkpoint
        filename = Path(os.path.join(args.exp_dir, f"epoch-{epoch}.pt"))
        save_checkpoint(filename=filename, model=model, optimizer=optimizer, scheduler=scheduler, rank=rank)

    logging.info("Done!")

    if rank ==0 and args.wandb:
        wandb.finish()

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()

def main():
    args = get_parse()
    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()