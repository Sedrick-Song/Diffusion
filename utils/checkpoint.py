import torch
import logging
from torch.nn.parallel import DistributedDataParallel as DDP

# save ckpt

def save_checkpoint(filename, model, optimizer, scheduler, rank):
    if rank != 0:
        return None
    
    logging.info("Saving checkpoint to {}".format(filename))

    if isinstance(model, DDP):
        model = model.module
    
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None
    }

    torch.save(checkpoint, filename)
