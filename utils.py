# util functions, etc

import math

def get_custom_lr_scheduling_fn(linear_warmup_epochs, total_epochs):
    # create and return custom lr scheduling function that will be used for torch.optim.lr_schedule.LambdaLR
    # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.LambdaLR
    def lr_schedule_fn(epoch):
        # "use linear warmup for the first 10 epochs"
        if epoch < linear_warmup_epochs:
            return (epoch + 1) / linear_warmup_epochs

        # "decay the learning rate with the cosine decay schedule without restarts"
        # if eta_min is 0, and eta_max is initial lr, then lr at step t is equal to
        # initial_lr * (1/2) * (1 + cos(T_cur/T_max * pi))
        T_max = total_epochs - linear_warmup_epochs
        T_cur = epoch - linear_warmup_epochs
        return 1 / 2 * (1 + math.cos(T_cur / T_max * math.pi))
    
    return lr_schedule_fn
        