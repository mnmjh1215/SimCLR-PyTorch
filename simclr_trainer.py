
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from torch.utils.tensorboard import SummaryWriter

from loss import NTCrossEntropyLoss
from utils import get_custom_lr_scheduling_fn, AverageMeter, ProgressMeter

import os


class SimCLRTrainer:
    def __init__(self, model, train_dataloader, # model and data
                 learning_rate=3e-3, weight_decay=1e-6, temperature=0.5, 
                 linear_warmup_epochs=10, total_epochs=350, # training hyperparameters
                 print_interval=100, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        self.device = device

        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, get_custom_lr_scheduling_fn(linear_warmup_epochs, total_epochs))
        
        self.train_dataloader = train_dataloader
        
        self.criterion = NTCrossEntropyLoss(temperature, device=device)
        
        self.curr_epoch = 0
        self.total_epochs = total_epochs
        self.curr_step = 0
        
        self.best_val_loss = float("inf")
        
        self.print_interval = print_interval
        self.writer = SummaryWriter()

    def train(self):
        # training process
        
        for epoch in range(self.curr_epoch, self.total_epochs):
            self.curr_epoch += 1
            print("Learning Rate:", self.scheduler.get_last_lr())
            
            loss_avg_meter = AverageMeter('Loss', ':.4e')
            acc_avg_meter = AverageMeter('Acc', ':6.2f')
            time_avg_meter = AverageMeter('Time', ':6.3f')
            progress_meter = ProgressMeter(len(self.train_dataloader), 
                                           [time_avg_meter, loss_avg_meter, acc_avg_meter],
                                           prefix="Epoch: [{}]".format(epoch))
            start_time = time.time()
            for ix, ((x_i, x_j), _) in enumerate(self.train_dataloader):
                self.curr_step += 1

                # run train step
                x_i, x_j = x_i.to(self.device), x_j.to(self.device)
                
                loss, correct = self.train_step(x_i, x_j)
                acc = correct / (len(x_i) + len(x_j))
                self.writer.add_scalar("Train/MinibatchLoss", loss, self.curr_step)
                self.writer.add_scalar("Train/MinibatchAcc", acc, self.curr_step)

                loss_avg_meter.update(loss, x_i.size(0) + x_j.size(0))  
                acc_avg_meter.update(acc, x_i.size(0) + x_j.size(0))     
                time_avg_meter.update(time.time() - start_time)         
                
                if ix % self.print_interval == 0:
                    progress_meter.display(ix)
            
            # end of an epoch, lr scheduler step
            self.scheduler.step()    
            
            # print result of an epoch
            progress_meter.display(len(self.train_dataloader))
            
            self.writer.add_scalar("Train/Loss", loss_avg_meter.avg, self.curr_epoch)
            self.writer.add_scalar("Train/Accuracy", acc_avg_meter.avg, self.curr_epoch)                
        
    def train_step(self, x_i, x_j):
        # train step using supervised method
        
        self.model.train()
        
        z_i = self.model(x_i)
        z_j = self.model(x_j)
        
        loss, pred, targets = self.criterion(z_i, z_j)
        
        correct = torch.sum(pred == targets).item()
                
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
                
        return loss.item(), correct
                
    def save_checkpoint(self, checkpoint_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.curr_epoch,
            'step': self.curr_step
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.curr_epoch = checkpoint['epoch']
        self.curr_step = checkpoint['step']


