import torch
import logging
import wandb
from pathlib import Path
import os
from utils.checkpoint import save_checkpoint

#Training pipeline

def compute_loss(encodec, model, batch, device, is_training):
    waveforms, spectrogram, audio_mask, mel_mask = batch[0], batch[1], batch[2], batch[3]  # wavform B*T
    mel_mask = mel_mask.unsqueeze(1)
    waveforms = waveforms.unsqueeze(1) # waveform B*1*T
    waveforms = waveforms.to(device)
    discrete_token = encodec.encode(waveforms)
    latent_embedding = encodec.get_latent_embedding(discrete_token)
    spectrogram = spectrogram.to(device)
    latent_embedding = latent_embedding.permute(0,2,1).to(device)
    audio_mask = audio_mask.to(device)
    mel_mask = mel_mask.permute(0,1,3,2).to(device)
    with torch.set_grad_enabled(is_training):
        loss = model(latents=spectrogram, encoder_hidden_states=latent_embedding, boolean_encoder_mask=audio_mask, mel_mask=mel_mask, validation_mode=not is_training)
    return loss

def compute_validation_loss(encodec, model, valid_dl, device, world_size):
    model.eval()
    total_loss = 0.0
    total_length = len(valid_dl)
    for batch_idx, batch in enumerate(valid_dl):
        loss = compute_loss(encodec=encodec, model=model, batch=batch, device=device, is_training=False)
        assert loss.requires_grad is False
        total_loss += loss
    total_loss /= total_length
    return total_loss
    

def train_one_epoch(encodec, model, optimizer, scheduler, train_dl, valid_dl, epoch, args, device, world_size=1, rank=0):
    model.train()
    total_length = len(train_dl)
    total_loss = 0.0
    for batch_idx, batch in enumerate(train_dl):
        loss = compute_loss(encodec=encodec, model=model, batch=batch, device=device, is_training=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss
        total_loss_per_batch = total_loss / (batch_idx + 1)
        if (batch_idx % 100 == 0):
            cur_lr = max(scheduler.get_last_lr())
            logging.info(
                f"Epoch {epoch},"
                f"batch {batch_idx}, loss: {loss},"
                f"lr: {cur_lr:.2e}"
            )
            if args.wandb and rank==0:
                wandb.log({"train_loss":loss}, step=((epoch-1)*total_length+batch_idx))
        
        if (batch_idx % 1000 == 0):
            torch.cuda.empty_cache()

        if (batch_idx % 10000 == 0 and batch_idx != 0):
            filename = Path(os.path.join(args.exp_dir, f"epoch-{epoch}-checkpoint-{batch_idx}.pt"))
            save_checkpoint(filename=filename, model=model, optimizer=optimizer, scheduler=scheduler, rank=rank)
        '''
        if (batch_idx % 3000 == 0) and (batch_idx != 0):
            logging.info("Computing validation loss")
            valid_loss = compute_validation_loss(encodec=encodec, model=model, valid_dl=valid_dl, device=device, world_size=world_size)
            model.train()
            logging.info(f"Epoch {epoch} batch {batch_idx}, validation loss: {valid_loss}")
            if args.wandb and rank==0:
                wandb.log({"valid_loss":valid_loss}, step=((epoch-1)*total_length+batch_idx))
        '''
        
    

            
