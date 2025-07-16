import argparse 
from typing_extensions import Literal # python>=3.8用typing就行
from typing import List, Optional
from pathlib import Path 
import os 
from multiprocessing import Pool

from tqdm import tqdm 
import pandas as pd 
import numpy as np 
import scipy.sparse as sps

import torch
from torch.utils.data import Dataset
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
from fragment_mol.utils.fingerprint import get_batch_fp, get_batch_fp_multiprocess
from fragment_mol.models.model_utils import positional_encoding
from fragment_mol.dove.mol_bpe_new import TokenizerNew, MolGraph

__all__ = ["FragGraphDataset", "FragGraphCollator"]

class Smiles2Fragment(object):
    def __init__(self, order=1, vocab_size=500, save_to_local=False, max_length=2) -> None:
        # super().__init__()
        self.save_to_local = save_to_local
        vocab_path = f"/mnt/nvme_share/wangjx/gnn_tools/fragment_mol/ps_lg/chembl29_vocab_new_lg{order}_{vocab_size}.txt"
        print(f"load vocab from {vocab_path}")
        # self.tokenizer = TokenizerNew(vocab_path=f"/mnt/nvme_share/wangjx/gnn_tools/fragment_mol/ps_lg/chembl29_vocab_lg{order}_{vocab_size}.txt",
        #                               order=order)
        self.tokenizer = TokenizerNew(vocab_path=vocab_path, order=order)
        # self.tokenizer = TokenizerNew(vocab_path="/mnt/nvme_share/wangjx/gnn_tools/fragment_mol/ps_lg/chembl29_vocab_lg.txt",
        #                               order=order)
        self.max_length = max_length
        
    
    def sep_group_idx(self, group_idx):
        node_ids = [] 
        macro_node_ids = [] # 
        for i, group in enumerate(group_idx):
            macro_node_ids.extend([i]*len(group))
            node_ids.extend(group)
        return torch.LongTensor(node_ids), torch.LongTensor(macro_node_ids)
    
    def smiles2fragment(self, smiles, max_length=None):  
        if max_length is None:
            max_length = self.max_length
        mol = Chem.MolFromSmiles(smiles)
        # build vanilla graph
        node_features = [atom_featurizer_all(atom) for atom in mol.GetAtoms()]
        edge_features = []
        edges = []
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.append((u, v))
            edges.append((v, u))
            
            edge_features.extend([bond_featurizer_all(bond)]*2)
        
        if len(edges) == 0:
            print(f"{smiles} contains no bond!")
            src, tgt = [0], [0]
            edge_features.append([0 for _ in range(BOND_FEATURE_DIM)])
        else:
            src, tgt = list(zip(*edges))
        vanilla_graph = dgl.graph((src, tgt), num_nodes=mol.GetNumAtoms())
        vanilla_graph.ndata['h'] = torch.FloatTensor(node_features)
        vanilla_graph.edata['e'] = torch.FloatTensor(edge_features)
        
        # build fragment group_idxs, fragment_idxs
        mol = MolGraph(mol)
        group_idxs, fragment_idxs, fragment_idxs_detail = self.tokenizer.tokenize(mol)
        # torch_scatter + group_idxs得到fragment_subgraph_embedding
        # fragment_idxs得到fragment_embedding
        node_labels = torch.stack([self.bit_to_idx(bits, len(self.tokenizer)) for bits in fragment_idxs_detail])
        # build fragformer connection graph
        mol.fragment_graph_transform()
        mol.complete_graph_transform(max_length=max_length)
        fragformer_edges = mol.edges
        fragformer_distance = mol.distance
        fragformer_paths = mol.paths
        
        if len(fragformer_edges) == 0:
            print(f"{smiles} contains no bond in fragment graph!") # 理论上不应该进入这里
            print(group_idxs, len(group_idxs))
            print(fragformer_edges, fragformer_distance, fragformer_paths)
            src, tgt = [0], [0]
            # edge_features.append([0 for _ in range(BOND_FEATURE_DIM)])
        else:
            src, tgt = list(zip(*fragformer_edges))
            
        fragformer_graph = dgl.graph((src, tgt), num_nodes=len(group_idxs))
        fragformer_graph.edata['distance'] = torch.LongTensor(fragformer_distance)
        fragformer_graph.edata['paths'] = torch.LongTensor(fragformer_paths)
        fragformer_graph.ndata['id'] = torch.LongTensor(fragment_idxs)
        fragformer_graph.ndata['label'] = node_labels
        node_ids, macro_node_ids = self.sep_group_idx(group_idx=group_idxs)
        return {
            'vanilla_graph': vanilla_graph,
            'group_idxs': group_idxs,
            'node_ids': node_ids, 
            'macro_node_ids': macro_node_ids,
            'fragformer_graph': fragformer_graph,
        }
        # return vanilla_graph, group_idxs, fragformer_graph
    def bit_to_idx(self, bits, vocab_size):
        label = torch.zeros(vocab_size)
        label[bits] = 1
        return label
    
    def __call__(self, smiles):
        # print(smiles)
        if type(smiles) == tuple:
            smiles, path = smiles
        # if path.exists() and self.save_to_local:
        #     return 
        r = self.smiles2fragment(smiles)
        if self.save_to_local:
            # path = f"/ssd/wangjx/mol_repre/KPGT/datasets/chembl29/cached_graphs/{index}.pt"
            torch.save(r, path)
        else:
            return r

def normalize_target(target, mean, std):
    target = (np.array(target)-mean)/std
    return target.tolist()

@register_dataset('frag_graph')
class FragGraphDataset(Dataset):
    # 不确定是否有必要继承Dataset类, 继承了没错
    def __init__(self, dataset: str, 
                 split: Literal["train", 'valid', 'test']='train', 
                 scaffold_id: int=0,
                 args: Optional[argparse.Namespace]=None) -> None:
        # self.args = args
        benchmark_name = BENCHMARK_NAME[dataset]
        self.dataset = dataset
        self.benchmark_name = benchmark_name
        base_path = BASE_PATH[benchmark_name]
        # base_path = Path("/mnt/nvme_share/wangjx/mol_repre/KPGT/datasets")
        descriptor_dir = base_path / dataset / "descriptor"
        if not descriptor_dir.exists():
            print("create descriptor dir")
            descriptor_dir.mkdir(parents=True)
        # ecfp_path = base_path / dataset / "rdkfp1-7_512.npz"
        # md_path = base_path / dataset / "molecular_descriptors.npz"
        dataset_file = base_path / dataset / f"{dataset}.csv"
        df = pd.read_csv(dataset_file)
        self.knodes = {}
        for fp_name in args.knodes:
            fp_path = descriptor_dir / f"{fp_name}.npz"
            if not fp_path.exists():
                print(f"create {fp_name} descriptor")
                # fp = get_batch_fp(df["smiles"].values.tolist(), fp_name)
                fp = get_batch_fp_multiprocess(df["smiles"].values.tolist(), fp_name)
                if (fp != fp).sum()>0:
                    print(f"{(fp != fp).sum()} nan in {fp_name} descriptor, fill with 0")
                    fp = np.nan_to_num(fp, nan=0)
                np.savez(fp_path, fp=fp)
            else:
                print(f"load {fp_name} descriptor")
                fp = np.load(fp_path, allow_pickle=True)['fp']
                if (fp != fp).sum()>0:
                    print(f"{(fp != fp).sum()} nan in {fp_name} descriptor, fill with 0")
                    fp = np.nan_to_num(fp, nan=0)
            self.knodes[fp_name] = torch.from_numpy(fp.astype(np.float32))
        # fps = torch.from_numpy(sps.load_npz(ecfp_path).todense().astype(np.float32))
        # mds = np.load(md_path)['md'].astype(np.float32)
        # mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))
        split_name = get_split_name(dataset=dataset, scaffold_id=scaffold_id)
        # split_path = base_path / dataset / "splits" / f"scaffold-{scaffold_id}.npy"
        split_path = base_path / dataset / "splits" / f"{split_name}.npy"
        split_dict = np.load(split_path, allow_pickle=True)
        use_idxs = split_dict[SPLIT_TO_ID[split]]
        df = df.iloc[use_idxs]
        # self.fps, self.mds = fps[use_idxs], mds[use_idxs]
        self.smiles = df["smiles"].values.tolist()
        self.knodes = {fp_name: self.knodes[fp_name][use_idxs] for fp_name in self.knodes}
        df.drop("smiles", axis=1, inplace=True)
        # compute positive rate of each column
        print(df.mean(axis=0))
        self.targets = df.values.tolist() # 剩余列都是label, muiti-class classification / regression
        self.num_targets = len(self.targets[0])
        # n_virtual_nodes = 2 if args.knode else 0
        self.frag_helper = Smiles2Fragment(order=args.order, vocab_size=args.vocab_size)
        # cache_data_dir = Path("/mnt/nvme_share/wangjx/gnn_tools/fragment_mol/cached_data") / "frag_graph"
        
        cache_data_dir = Path("/ssd/wangjx/cached_data") / "frag_graph"
        cache_data_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = cache_data_dir / f"{dataset}_order{args.order}_{args.vocab_size}_{split}_{scaffold_id}.pt"
        if dataset_path.exists():
            print(f"load cached dataset from {dataset_path}")
            self.graphs = torch.load(dataset_path)
        else:
            self.graphs = [self.frag_helper(s) for s in tqdm(self.smiles)]
            # n_jobs = 32
            # for i in range(11):
            #     local_smiles = self.smiles[i*1000:(i+1)*1000]
            #     graphs = list(Pool(n_jobs).imap(self.frag_helper, tqdm(local_smiles)))
            #     torch.save(graphs, str(dataset_path)+f"{i}")
            # exit()
            # self.graphs = list(Pool(n_jobs, maxtasksperchild=10).imap(self.frag_helper, tqdm(self.smiles)))
            print(f"save cached graph dataset to {dataset_path}")
            torch.save(self.graphs, dataset_path)
        
        index_list = [i for i in range(len(self.graphs)) if self.graphs[i] is not None] 
        # print("max index", max(index_list))
        print(f"# of None in {split} split: {len(self.graphs)-len(index_list)}")
        self.graphs = [self.graphs[i] for i in index_list]
        # for graph in self.graphs:
        #     graph.ndata['pe'] = positional_encoding(graph['vanilla_graph], target_dim=6, pe_type=args.pe_type)
        # self.node_indicators = [torch.LongTensor(self.node_indicators[i]) for i in index_list]
        self.targets = [self.targets[i] for i in index_list]
        self.knodes = {fp_name: self.knodes[fp_name][index_list] for fp_name in self.knodes}
        task_type = get_task_type(dataset)
        # print(task_type)
        if task_type == 'reg':
            print("normalize target for regression task")
            # print(split)
            if split == 'train' or (split == 'test' and args.debug):
                self.set_mean_and_std(args)
            # print(args)
            
            # self.targets = [normalize_target(target, args.target_mean, args.target_std) for target in self.targets]
        # self.fps = self.fps[index_list]
        # self.mds = self.mds[index_list]
    
    
    
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx], {fp_name: self.knodes[fp_name][idx] for fp_name in self.knodes}
    
    def set_mean_and_std(self, args):
        # if mean is None:
        #     mean = torch.from_numpy(np.nanmean(self.targets.numpy(), axis=0))
        # if std is None:
        #     std = torch.from_numpy(np.nanstd(self.targets.numpy(), axis=0))
        mean = np.nanmean(np.array(self.targets), axis=0)
        std = np.nanstd(np.array(self.targets), axis=0)
        # args.target_mean = torch.from_numpy(mean)
        # args.target_std = torch.from_numpy(std)
        args.target_mean = mean
        args.target_std = std
        print(f"target mean={mean}, std={std}")
        # print(args)
        
    def remove_negative_samples(self):
        assert self.benchmark_name == 'xai', f"wrong call of remove neg, {self.dataset}, {self.benchmark_name}"
        index_list = [i for i in range(len(self.targets)) if self.targets[i][0]==1]
        self.smiles = [self.smiles[i] for i in index_list]
        self.graphs = [self.graphs[i] for i in index_list]
        self.targets = [self.targets[i] for i in index_list]
        self.knodes = {fp_name: self.knodes[fp_name][index_list] for fp_name in self.knodes}
        self.pos_index_list = index_list
    
@register_collator('frag_graph')
class FragGraphCollator(object):
    def __init__(self, device=0, mask_rate=0.0, pretrain=False) -> None:
        self.device = device
        self.mask_rate = mask_rate
        self.pretrain = pretrain
    
    
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
        mask = torch.bernoulli(torch.full((bg.num_nodes(),), mask_rate))
        mask = mask.bool()
        return mask 
    
    def __call__(self, item_list):
        graphs, labels, knodes = list(zip(*item_list))
        batched_vanilla_graph = dgl.batch([g['vanilla_graph'] for g in graphs])
        group_idx_list = [g['group_idxs'] for g in graphs]
        batched_fragformer_graph = dgl.batch([g['fragformer_graph'] for g in graphs])
        node_ids_list = [g['node_ids'] for g in graphs]
        batched_node_ids = self.node_id_shift(node_ids_list, batched_vanilla_graph)
        macro_node_ids_list = [g['macro_node_ids'] for g in graphs]    
        batched_macro_node_ids = self.node_id_shift(macro_node_ids_list, batched_fragformer_graph)
        # fps = torch.stack(fps, dim=0).reshape(len(labels), -1)
        # mds = torch.stack(mds, dim=0).reshape(len(labels), -1)
        batched_vanilla_graph = batched_vanilla_graph.to(self.device)
        batched_fragformer_graph = batched_fragformer_graph.to(self.device)
        labels = torch.FloatTensor(labels).to(self.device)
        batched_knodes = {}
        for fp_name in knodes[0]:
            batched_knodes[fp_name] = torch.stack([knode[fp_name] for knode in knodes], dim=0).to(self.device)
        
        batched_node_ids = batched_node_ids.to(self.device)
        batched_macro_node_ids = batched_macro_node_ids.to(self.device)
        fragment_mask = self.generate_mask(batched_fragformer_graph, self.mask_rate).to(self.device)
        # fps = fps.to(self.device)
        # mds = mds.to(self.device)
        data = {}
        data['vanilla_g'] = batched_vanilla_graph
        data['fragformer_g'] = batched_fragformer_graph
        data['batch_node_ids'] = batched_node_ids
        data['batch_macro_node_ids'] = batched_macro_node_ids
        data['knodes'] = batched_knodes
        data['fragment_mask'] = fragment_mask
        data['group_idx_list'] = group_idx_list
        
        # data['fp'] = fps
        # data['md'] = mds
        return data, labels



def test_dataset():
    parser = argparse.ArgumentParser(description='Fragment based model')
    parser.add_argument('--dataset', type=str, default="bbbp", help='dataset')
    parser.add_argument('--scaffold_id', type=int, default=0, help='scaffold id')
    parser.add_argument('--split', type=str, default="train", help='split')
    args = parser.parse_args()
    # print(type(args)) # <class 'argparse.Namespace'>
    dataset = args.dataset
    scaffold_id = args.scaffold_id
    base_path = Path("/mnt/nvme_share/wangjx/mol_repre/KPGT/datasets")
    dataset_file = base_path / dataset / f"{dataset}.csv"
    df = pd.read_csv(dataset_file)
    split_path = base_path / dataset / "splits" / f"scaffold-{scaffold_id}.npy"
    split_dict = np.load(split_path, allow_pickle=True)
    use_idxs = split_dict[SPLIT_TO_ID[args.split]]
    df = df.iloc[use_idxs]
    smiles = df["smiles"].values.tolist()
    print(len(smiles))
    df.drop("smiles", axis=1, inplace=True)
    target = df.values.tolist()
    print(target[:3])

def test_smiles2graph():
    smiles = "O1CC[C@@H](NC(=O)[C@@H](Cc2cc3cc(ccc3nc2N)-c2ccccc2C)C)CC1(C)C"
    # graph = smiles2graph(smiles)
    # print(graph)

if __name__ == "__main__":
    # dataset = GraphDataset("bbbp", "train")
    # dataset = GraphDataset("clintox", "train")
    test_dataset()
    
    