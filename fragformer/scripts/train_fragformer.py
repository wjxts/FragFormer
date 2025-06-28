import sys
sys.path.append('..')

from src.utils import set_random_seed
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import dgl
import numpy as np
import os
import random
from src.trainer.scheduler import PolynomialDecayLR
from src.trainer.pretrain_trainer import Trainer
import warnings
import wandb 
from time import time
from tqdm import tqdm 

from fragment_mol.register import MODEL_DICT, DATASET_DICT, COLLATOR_DICT, MODEL_ARG_FUNC_DICT
from fragment_mol.utils.fingerprint import FP_FUNC_DICT
from fragment_mol.ps_lg.mol_bpe_new import TokenizerNew
from fragment_mol.models.model_utils import model_n_params 

warnings.filterwarnings("ignore")
local_rank = int(os.environ['LOCAL_RANK'])

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(args):
  
    
    args.local_rank = local_rank
    args.save_path = f"{args.save_path}_{args.downsize}"
    os.makedirs(args.save_path, exist_ok=True)

    if local_rank == 0:
        print(args)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device('cuda', local_rank)
    set_random_seed(args.seed)
    # print(local_rank)

    print("Loading data...")
    start_time = time()

    vocab_path = f"/mnt/nvme_share/wangjx/gnn_tools/fragment_mol/ps_lg/chembl29_vocab_new_lg{args.order}_{args.vocab_size}.txt"
    tokenizer = TokenizerNew(vocab_path=vocab_path, order=args.order)
  
    collator = COLLATOR_DICT['frag_graph_pretrain'](args)
    train_dataset = DATASET_DICT['frag_graph_pretrain'](args)
    train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset), batch_size=args.batch_size//args.n_devices, num_workers=args.n_threads, worker_init_fn=seed_worker, drop_last=True, collate_fn=collator)
    end_time = time()
    print(f"Data loaded, time cost: {end_time - start_time:.2f} s")
    model_class = MODEL_DICT['fragformer']
    model_args = MODEL_ARG_FUNC_DICT['fragformer']()
    n_tasks = len(tokenizer)
    model = model_class(args=model_args, n_tasks=n_tasks).to(device)
    print(f"# of params (M): {model_n_params(model)/10**6:.3f}")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=2000, tot_updates=25000, lr=args.lr, end_lr=1e-9, power=1)
    if args.no_hier_loss:
        clf_loss_fn = CrossEntropyLoss(reduction='none')
    else:
        clf_loss_fn = BCEWithLogitsLoss(reduction='none')
    trainer = Trainer(args, optimizer, lr_scheduler, clf_loss_fn, device=device, local_rank=local_rank)
    n_epochs = args.n_steps // len(train_loader) + 1
    trainer.fit(model, train_loader, n_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for training FragFormer")
    parser.add_argument("--seed", type=int, default=22)
    
    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--order", type=int, default=1, help="order of L(G)")
    parser.add_argument("--vocab_size", type=int, default=500, help="vocab size of fragment library")
    
    parser.add_argument("--n_steps", type=int, required=True)
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--n_devices", type=int, default=1)
    
    parser.add_argument('--lr', type=float, default=2e-04, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    
    
    parser.add_argument("--downsize", type=int, default=-1)
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--knodes", type=str, default=[], nargs="*", help="knowledge type",
                        choices=list(FP_FUNC_DICT.keys()))
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb')
    
    parser.add_argument('--no_hier_loss', action='store_true', help='whether to use hier loss')
    
    
    args, _ = parser.parse_known_args()
    
    if args.wandb and local_rank == 0:
        wandb.init(
                project = "FragFormer",
                name = f"Pretrain-{args.downsize}",
                config = args,
            )
    train(args=args)
        
    if args.wandb and local_rank == 0:
        wandb.finish()

