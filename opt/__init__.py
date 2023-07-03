import torch

def warmup(current_step: int, warmup_steps:int, training_steps:int):
    if current_step < warmup_steps:
        # current_step / warmup_steps * base_lr
        return float(current_step / warmup_steps)
    else:                                 
        # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
        return max(0.0, float(training_steps - current_step) / float(max(1, training_steps - warmup_steps)))

def get_warm_up(optimizer, current_step: int, warmup_steps:int, training_steps:int):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
