import torch
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm 
import wandb 
from time import time 

class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, clf_loss_fn, device, ddp=False, local_rank=1):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.clf_loss_fn = clf_loss_fn
        self.device = device
        self.ddp = ddp
        self.local_rank = local_rank
        self.n_updates = 0
        self.time_cost = {'load_data': 0, 'model_training': 0}
    
    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        tik = time()
        for batch_idx, batched_data in enumerate(train_loader):
            tok = time()
            # print(f"load data cost: {tok-tik:.2f}s, rank={self.local_rank}")
            self.time_cost['load_data'] += tok-tik
            tik = time()
            
            # move to cuda
            for key in batched_data:
                if key == 'knodes':
                    batched_data[key] = {k: v.to(self.device) for k, v in batched_data[key].items()}
                else:
                    batched_data[key] = batched_data[key].to(self.device)
                    
            self.optimizer.zero_grad()
            
            mask = batched_data['fragment_mask']
            label = batched_data['fragformer_g'].ndata['label']
            logits = model(batched_data, pretrain=True)
            # print(logits.shape, label.shape, mask.shape)
            # sl_predictions, sl_labels, fp_predictions, fps, md_predictions, mds = self._forward_epoch(model, batched_data)
            # print(f"mask shape: {mask.shape}")
            if self.args.no_hier_loss:
                label = label.reshape(-1)
            loss = (self.clf_loss_fn(logits[mask], label[mask])).mean()
            # sl_labels = sl_labels.reshape(-1)
            # fps = fps.float()
            # sl_loss = self.sl_loss_fn(sl_predictions, sl_labels).mean()
            # fp_loss = self.clf_loss_fn(fp_predictions, fps).mean()
            # md_loss = self.reg_loss_fn(md_predictions, mds).mean()
            # loss = (sl_loss + fp_loss + md_loss)/3
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            self.optimizer.step()
            self.n_updates += 1
            self.lr_scheduler.step()
            d_loss = {"Loss": loss.item()}
            # print(d_loss)
            if self.args.wandb and self.local_rank == 0:
                # wandb.log有个step参数，可以用来指定记录的step
                wandb.log(d_loss)
            # if self.local_rank == 0:
            if self.n_updates % 100 == 1 and self.local_rank == 0:
                print(self.n_updates, d_loss)
            if self.n_updates % (self.args.n_steps // 10) == 1 and self.local_rank == 0:
                self.save_model(model, self.n_updates, epoch_idx)

            tok = time()
            
            self.time_cost['model_training'] += tok-tik
            if self.local_rank == 0 and self.n_updates % 10 == 1:
                print(f"load data cost: {self.time_cost['load_data']:.2f}s, model training cost: {self.time_cost['model_training']:.2f}s, ratio={self.time_cost['load_data']/self.time_cost['model_training']:.2f}")
            tik = time()
            
    def progress_wrapper(self):
        if self.local_rank==0:
            return tqdm 
        else:
            return lambda x: x 
        
    def fit(self, model, train_loader, n_epochs):
        if self.local_rank == 0:
            print(f"start training, # of epochs: {n_epochs}")
        
        wrapper = self.progress_wrapper()
        for epoch in wrapper(range(1, n_epochs+1)):
            model.train()
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)
            if self.n_updates >= self.args.n_steps:
                print("Training finished")
                break

    def save_model(self, model, n_steps, epoch):
        torch.save(model.state_dict(), self.args.save_path+f"/{self.args.config}_{epoch}_{n_steps}.pth")

    