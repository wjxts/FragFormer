import argparse 
from typing_extensions import Literal # python>=3.8用typing就行
from typing import List, Optional
from pathlib import Path 
import os 

from tqdm import tqdm 
import pandas as pd 
import numpy as np 
import scipy.sparse as sps

import torch
from torch.utils.data import Dataset, DataLoader
import dgl 
from rdkit import Chem

from fragment_mol.utils.chem_utils import bond_featurizer_all, atom_featurizer_all, get_split_name


from fragment_mol.utils.chem_utils import (ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
                              SPLIT_TO_ID, 
                              VIRTUAL_ATOM_FEATURE_PLACEHOLDER, # not used in fact, only for placeholder
                              VIRTUAL_BOND_FEATURE_PLACEHOLDER,
                              BENCHMARK_NAME,
                              BASE_PATH,
                              get_task_type
                              )

from fragment_mol.register import register_dataset, register_collator
from fragment_mol.utils.fingerprint import get_batch_fp, FP_DIM
from fragment_mol.models.model_utils import positional_encoding
from fragment_mol.ps_lg.mol_bpe_new import TokenizerNew, MolGraph
from fragment_mol.datasets.dataset_frag_graph import Smiles2Fragment

import lmdb
import io 
from time import time

__all__ = ["FragGraphPretrainDataset", "FragGraphPretrainCollator"]

@register_dataset('frag_graph_pretrain')
class FragGraphPretrainDataset(Dataset):
    # 不确定是否有必要继承Dataset类, 继承了没错
    def __init__(self, args) -> None:
        # self.args = args
        self.downsize = args.downsize
        self.dataset_size = 2084723 if self.downsize<0 else self.downsize
        dataset = 'chembl29'
        benchmark_name = BENCHMARK_NAME[dataset]
        base_path = BASE_PATH[benchmark_name]
        # base_path = Path("/mnt/nvme_share/wangjx/mol_repre/KPGT/datasets")
        descriptor_dir = base_path / dataset / "descriptor"
        # dataset_file = base_path / dataset / f"{dataset}.csv"
        # df = pd.read_csv(dataset_file)
        self.knodes = {}
        for fp_name in args.knodes:
            # if self.downsize>0 and self.downsize<=50000:
            #     fp_path = descriptor_dir / f"{fp_name}_50000.npz"
            #     size = 50000
            # else:
            #     fp_path = descriptor_dir / f"{fp_name}.npz"
            #     size = 2084723
            fp_path = descriptor_dir / f"{fp_name}.npy"
            print(f"load {fp_name} descriptor from {fp_path}")
            begin_time = time()
            # fp = np.load(fp_path)['fp']
            # fp_dim = FP_DIM[fp_name]
            # fp = np.memmap(fp_path, dtype='float32', mode='r', shape=(size, fp_dim)) # 处理大特征矩阵的方法
            fp = np.load(fp_path, mmap_mode='r') 
            end_time = time()
            print(f"load {fp_name} time cost: {end_time-begin_time:.2f}s")
            # if (fp != fp).sum()>0:
            #     # 理论上不会进入这里
            #     print(f"{(fp != fp).sum()} nan in {fp_name} descriptor, fill with 0")
            #     fp = np.nan_to_num(fp, nan=0)
            # self.knodes[fp_name] = torch.from_numpy(fp.astype(np.float32))
            self.knodes[fp_name] = fp
        
        if args.order == 0:
            # self.lmdb_path = f'/mnt/nvme_share/wangjx/lmdb_order0_{args.vocab_size}_all'
            self.lmdb_path = f'/mnt/nvme_share/wangjx/lmdb_graph_order0_{args.vocab_size}'
            # self.lmdb_path = f'/ssd/wangjx/lmdb_graph_order0_{args.vocab_size}'
        elif args.order == 1:
            # self.lmdb_path = f'/mnt/nfs_share/wangjx/lmdb_order1_{args.vocab_size}_all'
            # self.lmdb_path = f'/mnt/nvme_share/wangjx/lmdb_order1_{args.vocab_size}_all'
            self.lmdb_path = f'/mnt/nvme_share/wangjx/lmdb_graph_order1_{args.vocab_size}'
            # self.lmdb_path = f'/ssd/wangjx/lmdb_graph_order1_{args.vocab_size}'
        elif args.order == 2:
            # self.lmdb_path = '/home/past/wangjx/lmdb_order2_500_all' 
            # self.lmdb_path = f'/home/wangjx/lmdb_order2_{args.vocab_size}_all' 
            self.lmdb_path = f'/mnt/nvme_share/wangjx/lmdb_graph_order2_{args.vocab_size}'
            assert os.path.exists(self.lmdb_path), f"{self.lmdb_path} not exists"
        elif args.order == 3:
            self.lmdb_path = f'/mnt/nvme_share/wangjx/lmdb_graph_order3_{args.vocab_size}'
            # self.lmdb_path = f'/ssd/wangjx/lmdb_graph_order3_{args.vocab_size}'
        else:
            raise ValueError(f"order {args.order} not supported")
        print(f"use data from {self.lmdb_path}")
        # self.lmdb_path = '/home/past/wangjx/lmdb_order2_500_all' # only for server #40
        # self.lmdb_path = '/mnt/nvme_share/wangjx/lmdb_order1_1000_all'
        self.env = lmdb.open(self.lmdb_path, readonly=True,
                             lock=False, readahead=False, meminit=False)
        # 加后三个变量可以避免env读太多数据了炸掉
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # graph_file = self.cached_graph_dir / f"{idx}.pt"
        # graph = torch.load(graph_file)
        # graph = self.load_single_data(idx)
        with self.env.begin() as txn:
            data_bin = txn.get(f"{idx}.pt".encode('utf-8'))
            graph = torch.load(io.BytesIO(data_bin))  # 需要使用 BytesIO 从字节流读取
            
        local_knodes = {}
        for fp_name in self.knodes:
            fp = self.knodes[fp_name][idx]
            fp_arr = np.array(fp).astype(np.float32)
            fp_arr = np.nan_to_num(fp_arr, nan=0) # 如果有nan，后边loss会变为nan
            local_knodes[fp_name] = torch.from_numpy(fp_arr)
        
        return graph, local_knodes
        # return graph, {fp_name: self.knodes[fp_name][idx] for fp_name in self.knodes}
    
    # def load_single_data(self, idx):
    #     env = lmdb.open(self.lmdb_path, readonly=True)
    #     with env.begin() as txn:
    #         data_bin = txn.get(f"{idx}.pt".encode('utf-8'))
    #         data = torch.load(io.BytesIO(data_bin))  # 需要使用 BytesIO 从字节流读取
    #     env.close()
    #     return data
    
    def __del__(self):
        # 关闭LMDB环境
        self.env.close()
            
    

@register_collator('frag_graph_pretrain')
class FragGraphPretrainCollator(object):
    def __init__(self, args) -> None:
        self.mask_rate = args.mask_rate
        self.no_hier_loss = args.no_hier_loss
    
    def node_id_shift(self, id_list, bg):
        # id_list: a list of 1-D torch.LongTensor
        # 获取每个图的节点数
        num_nodes_per_graph = bg.batch_num_nodes()
        accum_num_nodes = torch.cat([torch.tensor([0], device=num_nodes_per_graph.device), 
                                        num_nodes_per_graph.cumsum(0)[:-1]])
        assert len(id_list) == len(accum_num_nodes), "id_list and accum_num_nodes should have the same length"
        # print(num_nodes_per_fragment_graph.cumsum(0))
        # 为每个图创建一个与节点数相同长度的tensor，填充值为图的
        shift_id_list = [id_list[i]+accum_num_nodes[i] for i in range(len(id_list))]
        # 拼接所有的tensor得到所有节点的batch_id
        batch_shift_id = torch.cat(shift_id_list)
        return batch_shift_id
    
    def generate_mask(self, bg, mask_rate):
        # mask = torch.randint(0, 2, (bg.num_edges(),), device=bg.device)
        # mask = mask.bool()
        mask = torch.bernoulli(torch.full((bg.num_nodes(),), mask_rate, device=bg.device))
        mask = mask.bool()
        return mask 
    
    def hier_label2no_hier(self, x: torch.tensor):
        # x is a 2-D tensor
        return torch.tensor([torch.where(v)[0].max() for v in x]).reshape([-1, 1]) # 需要是整型
        
    def __call__(self, item_list):
        graphs, knodes = list(zip(*item_list))
        if self.no_hier_loss:
            for g in graphs:
                g['fragformer_graph'].ndata['label'] = self.hier_label2no_hier(g['fragformer_graph'].ndata['label'])
        batched_vanilla_graph = dgl.batch([g['vanilla_graph'] for g in graphs])
        batched_fragformer_graph = dgl.batch([g['fragformer_graph'] for g in graphs])
        node_ids_list = [g['node_ids'] for g in graphs]
        batched_node_ids = self.node_id_shift(node_ids_list, batched_vanilla_graph)
        macro_node_ids_list = [g['macro_node_ids'] for g in graphs]    
        batched_macro_node_ids = self.node_id_shift(macro_node_ids_list, batched_fragformer_graph)
        
        batched_knodes = {}
        for fp_name in knodes[0]:
            batched_knodes[fp_name] = torch.stack([knode[fp_name] for knode in knodes], dim=0)

        fragment_mask = self.generate_mask(batched_fragformer_graph, self.mask_rate)
  
        data = {}
        data['vanilla_g'] = batched_vanilla_graph
        data['fragformer_g'] = batched_fragformer_graph
        data['batch_node_ids'] = batched_node_ids
        data['batch_macro_node_ids'] = batched_macro_node_ids
        data['knodes'] = batched_knodes
        data['fragment_mask'] = fragment_mask
        
        return data
    
if __name__ == "__main__":
    pass
    
    