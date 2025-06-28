import sys
sys.path.append('..')

from src.utils import set_random_seed
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss
import numpy as np
import random
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from fragment_mol.utils.chem_utils import DATASET_TASKS
from fragment_mol.utils.utils import WarmUpLR
from fragment_mol.models.model_utils import ModelWithEMA 

from src.trainer.scheduler import PolynomialDecayLR
from src.trainer.finetune_trainer import Trainer
from fragment_mol.evaluator import Evaluator
import time 
import json 
from pathlib import Path 
import wandb 

from fragment_mol.register import MODEL_DICT, DATASET_DICT, COLLATOR_DICT, MODEL_ARG_FUNC_DICT
from fragment_mol.utils.chem_utils import DATASET_TASKS, get_task_metrics, get_task_type, METRIC_BEST_TYPE 
from fragment_mol.utils.fingerprint import FP_FUNC_DICT
from fragment_mol.ps_lg.mol_bpe_new import TokenizerNew
from fragment_mol.models.model_utils import model_n_params 


import warnings
warnings.filterwarnings("ignore")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def valid_model(model, valid_dataloader, evaluators, criterion, args):
    model.eval()
    label_list = []
    predict_list = []
    epoch_loss = 0
    for idx, (input_data, labels) in enumerate(valid_dataloader):
        # input_data is smiles idx for smiles
        # input_data is dgl.Graph for graph / fragment_graph
        predict = model(input_data)
        label_list.append(labels)
        predict_list.append(predict)
        
        is_labeled = (~torch.isnan(labels)).to(torch.float32)
        labels = torch.nan_to_num(labels, nan=0.0)
        loss = (criterion(predict, labels) * is_labeled).mean()
        # loss = criterion(predict, labels)
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss/len(valid_dataloader)
    labels = torch.cat(label_list, dim=0)
    predicts = torch.cat(predict_list, dim=0)
    # print(type(labels), type(predicts)) # <class 'torch.Tensor'> <class 'torch.Tensor'>  
    # score = evaluator.eval(labels, predicts)
    score = {metric: evaluator.eval(labels, predicts) for metric, evaluator in evaluators.items()}
    return score, avg_loss

def finetune(args):
    set_random_seed(args.seed)
    
    dataset_class = DATASET_DICT['frag_graph']
    train_split = 'train' if not args.debug else 'test'
    train_dataset = dataset_class(args.dataset, split=train_split, scaffold_id=args.scaffold_id, args=args)
    valid_dataset = dataset_class(args.dataset, split="valid", scaffold_id=args.scaffold_id, args=args)
    test_dataset = dataset_class(args.dataset, split="test", scaffold_id=args.scaffold_id, args=args)
    
    collator_class = COLLATOR_DICT['frag_graph']
    collator = collator_class(device=args.device)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    n_tasks_ft = DATASET_TASKS[args.dataset]
   
    vocab_path = f"chembl29_vocab_dove{args.order}_{args.vocab_size}.txt"
    tokenizer = TokenizerNew(vocab_path=vocab_path)
    model_class = MODEL_DICT['fragformer']
    model_args = MODEL_ARG_FUNC_DICT['fragformer']()
    n_tasks_pt = len(tokenizer)
    model = model_class(args=model_args, n_tasks=n_tasks_pt)
    
    load_dict = {k.replace('module.',''): v for k, v in torch.load(f'{args.model_path}').items()}
    if "fragment_idx_embed.weight" not in model.state_dict() and "fragment_idx_embed.weight" in load_dict:
        load_dict.pop("fragment_idx_embed.weight")
    model.load_state_dict(load_dict)
            
    model.init_ft_predictor(n_tasks_ft, args.dropout)
    model = model.to(args.device)
    model_ema = ModelWithEMA(model, decay=args.ema)
   
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print("model have {}M paramerters in total".format(sum(x.numel() for x in model.parameters())//int(10**6)))
    
    task_type = get_task_type(dataset=args.dataset)
    if task_type == 'cls':
        criterion = BCEWithLogitsLoss(reduction='none').to(args.device)
    else:
        criterion = MSELoss(reduction='none').to(args.device)
        
    metrics = get_task_metrics(args.dataset)
    evaluators = {metric: Evaluator(name=args.dataset, eval_metric=metric, n_tasks=DATASET_TASKS[args.dataset], mean=getattr(args, 'target_mean', None), std=getattr(args, 'target_std', None)) for metric in metrics}
    

    if args.warmup:
        scheduler = WarmUpLR(optimizer, 
                            total_iters=len(train_dataloader)*args.epochs, 
                            warmup_iters=len(train_dataloader)*args.warmup_epochs)
        
    scheduler = PolynomialDecayLR(optimizer, warmup_updates=args.epochs*len(train_dataset)//args.batch_size//10, tot_updates=args.epochs*len(train_dataset)//args.batch_size, lr=args.lr, end_lr=1e-9, power=1)
    
    train_score_list = []
    valid_score_list = []
    valid_ema_score_list = []
    test_score_list = []
    test_ema_score_list = []
    for epoch in range(1, args.epochs+1):
        train_score, train_loss = valid_model(model, train_dataloader, evaluators, criterion, args)
        valid_score, valid_loss = valid_model(model, valid_dataloader, evaluators, criterion, args)
        test_score, test_loss = valid_model(model, test_dataloader, evaluators, criterion, args)
        valid_ema_score, valid_ema_loss = valid_model(model_ema, valid_dataloader, evaluators, criterion, args)
        test_ema_score, test_ema_loss = valid_model(model_ema, test_dataloader, evaluators, criterion, args)
        print(train_score)
        print("Train score:", ", ".join([f"{k}={v:.3f}" for k, v in train_score.items()]))
        print("Valid score:", ", ".join([f"{k}={v:.3f}" for k, v in valid_score.items()]))
        print("Valid EMA score:", ", ".join([f"{k}={v:.3f}" for k, v in valid_ema_score.items()]))
        print("Test score:", ", ".join([f"{k}={v:.3f}" for k, v in test_score.items()]))
        print("Test EMA score:", ", ".join([f"{k}={v:.3f}" for k, v in test_ema_score.items()]))
        def add_prefix(d: dict, prefix: str):
            return {f"{prefix}_{k}":v for k, v in d.items()}
        if args.wandb:
            wandb.log(add_prefix(train_score, 'train'))
            wandb.log(add_prefix(valid_score, 'valid'))
            wandb.log(add_prefix(test_score, 'test'))
            wandb.log(add_prefix(test_ema_score, 'test_ema'))
            wandb.log({"train_loss": train_loss,
                     "valid_loss": valid_loss,
                     "valid_ema_loss": valid_ema_loss,
                     "test_loss": test_loss,
                     "test_ema_loss": test_ema_loss}
                      )
            
        train_score_list.append(train_score)
        valid_score_list.append(valid_score)
        valid_ema_score_list.append(valid_ema_score)
        test_score_list.append(test_score)
        test_ema_score_list.append(test_ema_score)
        model.train()
        epoch_loss = 0
        for idx, (input_data, labels) in enumerate(train_dataloader):
            
                
            optimizer.zero_grad()
            predict = model(input_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels, nan=0.0)
            if task_type == 'reg':
                mean, std = torch.from_numpy(args.target_mean).to(labels), torch.from_numpy(args.target_std).to(labels)
                labels = (labels-mean)/std
            loss = (criterion(predict, labels) * is_labeled).mean()
            
            # loss = criterion(predict, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model_ema.update_ema(model)
            scheduler.step() 
           
            epoch_loss += loss.item()
            if args.wandb:
                wandb.log({"train_step_loss": loss.item()})
            if idx%10==0:
                print(f"Epoch {epoch}, batch {idx}, loss {loss.item()}")
        print(f"Epoch {epoch}, loss {epoch_loss/len(train_dataloader)}")
        # save model
        if epoch == args.epochs:
            model_base_dir = Path("../models/finetune") / args.dataset
            model_base_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_base_dir / f"{epoch}.pt")
                
    
    d_result = {}
    if task_type == 'reg':
        print(f"normalize factor mean={args.target_mean}, std={args.target_std}")
    for metric in valid_score_list[0]:
        train_scores = [score[metric] for score in train_score_list]
        valid_scores = [score[metric] for score in valid_score_list]
        valid_ema_scores = [score[metric] for score in valid_ema_score_list]
        test_scores = [score[metric] for score in test_score_list]
        test_ema_scores = [score[metric] for score in test_ema_score_list]
        if METRIC_BEST_TYPE[metric] == 'max':
            max_index = np.argmax(valid_scores)
            max_index_ema = np.argmax(valid_ema_scores)
            d_result[f'max_train_{metric}'] = max(train_scores)
            d_result[f'max_valid_{metric}'] = max(valid_scores)
            d_result[f'max_valid_ema_{metric}'] = max(valid_ema_scores)
            d_result[f'max_test_{metric}'] = max(test_scores)
            d_result[f'max_test_ema_{metric}'] = max(test_ema_scores)
            d_result[f'final_test_{metric}'] = test_scores[max_index]
            d_result[f'final_test_ema_{metric}'] = test_ema_scores[max_index_ema]
            print((
                f"max train {metric}: {max(train_scores):.3f}, "
                f"max valid {metric}: {max(valid_scores):.3f}, "
                f"max valid ema {metric}: {max(valid_ema_scores):.3f}, "
                f"final test {metric}: {test_scores[max_index]:.3f}, "
                f"max test {metric}: {max(test_scores):.3f}, "
                f"final test ema {metric}: {test_ema_scores[max_index_ema]:.3f}, "
                f"max test ema {metric}: {max(test_ema_scores):.3f}, "
                ))
        else:
            min_index = np.argmin(valid_scores)
            d_result[f'min_train_{metric}'] = min(train_scores)
            d_result[f'min_valid_{metric}'] = min(valid_scores)
            d_result[f'min_valid_ema_{metric}'] = min(valid_ema_scores)
            d_result[f'min_test_{metric}'] = min(test_scores)
            d_result[f'min_test_ema_{metric}'] = min(test_ema_scores)
            d_result[f'final_test_{metric}'] = test_scores[min_index]
            d_result[f'final_test_ema_{metric}'] = test_ema_scores[min_index]
            print((
                f"min train {metric}: {min(train_scores):.3f}, "
                f"min valid {metric}: {min(valid_scores):.3f}, "
                f"min valid ema {metric}: {min(valid_ema_scores):.3f}, "
                f"final test {metric}: {test_scores[min_index]:.3f}, "
                f"min test {metric}: {min(test_scores):.3f}, "
                f"final test ema {metric}: {test_ema_scores[min_index]:.3f}, "
                f"min test ema {metric}: {min(test_ema_scores):.3f}, "
                ))

    if args.wandb:
        wandb.log(d_result)
  
    
    
    
    
if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Arguments for training LiGhT")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument('--device', type=int, default=0, help='device id')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=1, help='# of warmup epochs')
    parser.add_argument('--warmup', action='store_true', help='whether to use warmup')
    parser.add_argument("--ema", type=float, default=0.9, help='ema decay')
    parser.add_argument("--order", type=int, default=1, help="order of L(G)")
    parser.add_argument("--vocab_size", type=int, default=500, help="vocab size of fragment library")
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument('--scaffold_id', type=int, default=0, help='scaffold id')
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--knodes", type=str, default=[], nargs="*", help="knowledge type",
                        choices=list(FP_FUNC_DICT.keys()))
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb')
    parser.add_argument('--debug', action='store_true', help='debug mode') 
    
    args, _ = parser.parse_known_args()
    
    if args.wandb:
        wandb.init(
                project = "FragFormer",
                name = f"finetune-{args.dataset}-{args.scaffold_id}",
                config = args,
            )
    print(f"finetune on {args.dataset}, {args.scaffold_id}, model={args.model_path}")
    
    finetune(args)
    end_time = time.time()
    print(f"Time cost: {end_time-start_time:.2f}s")
    if args.wandb:
        wandb.finish()
    
    


    
    
    


