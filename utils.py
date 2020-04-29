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
        
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


