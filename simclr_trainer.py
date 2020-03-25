# trainer that implements base supervised learning algorithm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from torch.utils.tensorboard import SummaryWriter

from loss import NTCrossEntropyLoss
from utils import get_custom_lr_scheduling_fn

import os


class SimCLRTrainer:
    def __init__(self, model, train_dataloader, validation_dataloader, # model and data
                 learning_rate=3e-3, weight_decay=1e-6, temperature=0.5, 
                 linear_warmup_epochs=10, total_epochs=350, # training hyperparameters
                 print_interval=100, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        self.device = device

        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, get_custom_lr_scheduling_fn(linear_warmup_epochs, total_epochs))
        
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        
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
            
            epoch_loss = 0
            epoch_correct = 0
            epoch_count = 0
            start_time = time.time()
            for ix, ((x_i, x_j), _) in enumerate(self.train_dataloader):
                self.curr_step += 1

                # run train step
                x_i, x_j = x_i.to(self.device), x_j.to(self.device)
                
                loss, correct = self.train_step(x_i, x_j)
                self.writer.add_scalar("Train/MinibatchLoss", loss, self.curr_step)
                epoch_loss += loss
                epoch_correct += correct
                epoch_count += len(x_i) + len(x_j)
                
                if (ix + 1) % self.print_interval == 0:
                    print("Epoch [{0}/{1}] Step: {2} Loss: {3:.4f} time: {4:.2f}s".format(self.curr_epoch, self.total_epochs, (ix + 1),
                                                                                          epoch_loss / (ix + 1),
                                                                                          time.time() - start_time))
            
            # end of an epoch, lr scheduler step
            self.scheduler.step()    
            
            train_loss = epoch_loss / (ix + 1)
            train_acc = epoch_correct / epoch_count

            # print result of an epoch
            print("Epoch [{0}/{1}] Loss: {2:.4f}, Accuracy: {3:.4f}({4}/{5}) time: {6:.2f}s".format(self.curr_epoch, self.total_epochs, 
                                                                         train_loss, 
                                                                         train_acc,
                                                                         epoch_correct,
                                                                         epoch_count,
                                                                         time.time() - start_time))
            
            self.writer.add_scalar("Train/Loss", train_loss, self.curr_epoch)
            self.writer.add_scalar("Train/Accuracy", train_acc, self.curr_epoch)
                
            val_loss, val_acc = self.validate()
            print("Validation: Epoch [{0}/{1}] Validation Loss: {2:.4f}, Validation Accuracy: {3:.4f}".format(self.curr_epoch, self.total_epochs, 
                                                                                                    val_loss,
                                                                                                    val_acc))
            
            self.writer.add_scalar("Validation/Loss", val_loss, self.curr_epoch)
            self.writer.add_scalar("Validation/Accuracy", val_acc, self.curr_epoch)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

                if not os.path.isdir("checkpoints"):
                    os.makedirs("checkpoints")
                
                self.save_checkpoint("checkpoints/best_val_loss.pth")
                
        
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
            
    def validate(self):
        # validation step to calculate validation loss and validation accuracy
        val_loss_sum = 0
        val_correct = 0
        val_count = 0
        self.model.eval()
        with torch.no_grad():
            for ix, ((x_i, x_j), _) in enumerate(self.validation_dataloader):
                x_i, x_j = x_i.to(self.device), x_j.to(self.device)
                
                z_i = self.model(x_i)
                z_j = self.model(x_j)
                
                loss, pred, targets = self.criterion(z_i, z_j)
                
                val_loss_sum += loss.item() * 2 * len(x_i)
                val_correct += torch.sum(pred == targets).item()
                val_count += 2 * len(x_i)
                
        val_loss = val_loss_sum / val_count
        val_acc = val_correct / val_count
                
        return val_loss, val_acc
    
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


