#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
from copy import copy
import argparse
import multiprocessing as mp
from setuptools import sic
from tqdm import tqdm

from .utils.chem_utils import smi2mol, mol2smi, get_submol
from .utils.chem_utils import cnt_atom, CHEMBL_ATOMS
from .utils.logger import print_log
# from .molecule import Molecule

import networkx as nx
import numpy as np 
from itertools import permutations
from collections import defaultdict
'''classes below are used for principal subgraph extraction'''
INF = 10000000

class MolGraph:
    def __init__(self, mol) -> None:
        # atom, node, macro_node, edges
        self.mol = mol
        self.nodes = [{i} for i in range(self.mol.GetNumAtoms())] 
        # nodes[i]表示id为i的node包含的原子序号, 是一个set. 一个node就是一些原子的集合
        self.edges = [] # edge between nodes
        for bond in self.mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            # self.node_adj[u, v] = self.node_adj[v, u] = 1
            self.edges.append((u, v)) # 注意，这里存的是无向边
    
    @property
    def num_nodes(self):
        return len(self.nodes)
    
    def build_adj_mat(self):
        # 由edges生成邻接矩阵
        self.adj_mat = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int32)
        for u, v in self.edges:
            self.adj_mat[u, v] = self.adj_mat[v, u] = 1
    
    def build_adj_table(self):
        # 由edges生成邻接表
        self.adj_table = defaultdict(list)
        for u, v in self.edges:
            self.adj_table[u].append(v)
            self.adj_table[v].append(u)
    
    def build_init_macro_nodes(self):
        # 仅在分词时使用，在lining中不被使用
        self.macro_nodes_dict = {i: ([i], self.nodes[i]) for i in range(self.num_nodes)}
        self.nodeID_to_macroID = {i: i for i in range(self.num_nodes)}
        self.macro_nodes_id = self.num_nodes # 当前新的macro_node的id
        # 一个macro_node就是一个subgraph，用一个tuple表示，
        # 第一个元素是macro_node的node_id_list，第二个元素是macro_node包含的原子序号
    
    def update_graph(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
    
    def line_graph_transform(self):
        pairID_to_tripletID = -np.ones(shape=(self.num_nodes, self.num_nodes), dtype=np.int32)
        triplet_id = 0 # 记录新图当前的节点数(节点即tripelet)
        nodes = [] # 每个node是一个set，包含原子序数
        connected_nodes = set() # 与其他节点有连边的节点
        # 构建新图的节点，即将原图的边转化为新图的节点
        for i, (u, v) in enumerate(self.edges):
            # print(i, u, v)
            pairID_to_tripletID[u, v] = pairID_to_tripletID[v, u] = triplet_id
            nodes.append(self.nodes[u] | self.nodes[v]) # merge两个node的原子序数
            connected_nodes.add(u)
            connected_nodes.add(v)
            triplet_id += 1
        edges = []
        # 构建新图的边
        for i in range(self.num_nodes):
            node_ids = pairID_to_tripletID[i]
            node_ids = node_ids[node_ids>=0]
            if len(node_ids) >= 2:
                new_edges = list(permutations(node_ids, 2)) # 同一个节点对应的line是相连的
                # 这里包含了(u,v)和(v,u)两种情况，要去重
                new_edges = list(set([tuple(sorted(edge))for edge in new_edges]))
                # print(node_ids, new_edges)
                edges.extend(new_edges)
        
        # 处理孤立节点 (不动)
        for node_id in range(self.num_nodes):
            if node_id not in connected_nodes: 
                nodes.append(self.nodes[node_id])
                # VIRTUAL_NODE_ID在后续处理时可能有点麻烦，没法用于获取feature
                triplet_id += 1
                
        self.update_graph(nodes, edges)
    
    def remove_duplicate_nodes(self):
        # 去除重复的nodes
        new_nodes = []
        for node in self.nodes:
            if node not in new_nodes:
                new_nodes.append(node)
        new_edges = []
        for u, v in self.edges:
            if u < len(new_nodes) and v < len(new_nodes):
                new_edges.append((u, v))
        self.update_graph(new_nodes, new_edges)
        
    def fragment_graph_transform(self, group_idxs=None):
        if group_idxs is None:
            group_idxs = [i[1] for i in self.macro_nodes_dict.values()]
        num_fragments = len(group_idxs)
        new_edges = []
        for i in range(num_fragments):
            new_edges.append((i, i)) # add self-loop, does not influence the shortest path
        for i in range(num_fragments):
            group_atoms1 = group_idxs[i]
            for j in range(i+1, num_fragments):
                group_atoms2 = group_idxs[j]
                if len(group_atoms1 & group_atoms2) > 0: # 如果两个fragment的原子集合有交集，则连边
                    new_edges.append((i, j))
                    new_edges.append((j, i))
        self.edges = new_edges
                    
    def complete_graph_transform(self, max_length=5, add_self_loop=True):
        # 需要已经进行了对称化
        # 只需要self.edges
        # 生成新的self.edges和self.distance, self.paths
        nx_graph = nx.Graph(self.edges)
        if len(self.edges) == 0:
            nx_graph.add_nodes_from([1])
        # print(nx_graph)
        paths_dict = dict(nx.algorithms.all_pairs_shortest_path(nx_graph, max_length))
        # max_length是路径长度，路径上的节点个数最多是max_length+1
        new_edges, distance, paths = [], [], []
        for i in paths_dict:
            for j in paths_dict[i]:
                if i == j and not add_self_loop:
                    continue
                new_edges.append((i, j)) 
                distance.append([len(paths_dict[i][j])-1])
            
                paths.append(paths_dict[i][j]+[-INF]*(max_length+1-len(paths_dict[i][j])))
                
        # 每个原子自动会和自己相连
        self.distance = distance 
        self.paths = paths
        self.edges = new_edges
    
    def symmetrize_edges(self):
        # 使边是对称的
        new_edges = []
        for u, v in self.edges:
            new_edges.append((u, v))
            new_edges.append((v, u))
        self.edges = new_edges
    
    def get_nodes_smiles(self):
        cnt_smi = defaultdict(int)
        for node in self.nodes:
            submol = get_submol(self.mol, list(node))
            try:
                smi =  mol2smi(submol)
            except:
                continue
            cnt_smi[smi] += 1
        return cnt_smi
    
    def get_smis_subgraphs(self):
        res = []
        for _, atom_ids in self.macro_nodes_dict.values():
            submol = get_submol(self.mol, list(atom_ids))
            smi = mol2smi(submol)
            res.append((smi, atom_ids))
        return res
    
    def get_smis_subgraphs_new(self):
        res = []
        for key in self.macro_nodes_dict:
            _, atom_ids = self.macro_nodes_dict[key]
            detail_fragment_idx = self.fragment_idx_dict[key]
            submol = get_submol(self.mol, list(atom_ids))
            smi = mol2smi(submol)
            res.append((smi, atom_ids, detail_fragment_idx))
        # for _, atom_ids in self.macro_nodes_dict.values():
        #     submol = get_submol(self.mol, list(atom_ids))
        #     smi = mol2smi(submol)
        #     res.append((smi, atom_ids))
        return res
    
    def merge_macro_nodes(self, macro_node_id1, macro_node_id2):
        # return merged smiles
        atom_idxs = self.macro_nodes_dict[macro_node_id1][1] | self.macro_nodes_dict[macro_node_id2][1]
        submol = get_submol(self.mol, list(atom_idxs))
        smiles = mol2smi(submol)
        return smiles
    
    def init_macro_nodes_fragment_idx(self, tokenizer):
        self.fragment_idx_dict = defaultdict(list)
        for i, v in self.macro_nodes_dict.items():
            atomID_list = list(v[1])
            submol = get_submol(self.mol, atomID_list)
            smi =  mol2smi(submol)
            fragment_idx = tokenizer.smi_id(smi)
            self.fragment_idx_dict[i].append(fragment_idx)
    
    def get_nei_smis(self):
        # 统计将邻居节点合并后的smiles表达式个数
        cnt_smi = defaultdict(int)
        cnt_set = set()
        self.smi2macro_ids = defaultdict(list)
        for macro_node_id in self.macro_nodes_dict:
            # 对每个macro_node，找到其邻居macro_nodes，并合并
            node_ids, _ = self.macro_nodes_dict[macro_node_id]
            nei_macro_nodes = []
            for node_id in node_ids:
                for nei_id in self.adj_table[node_id]:
                    if nei_id not in node_ids:
                        nei_macro_nodes.append(self.nodeID_to_macroID[nei_id])
            for nei_macro_node in nei_macro_nodes:
                key = tuple(sorted((macro_node_id, nei_macro_node)))
                if key in cnt_set:
                    continue
                else:
                    cnt_set.add(key)
                    smi = self.merge_macro_nodes(macro_node_id, nei_macro_node)
                    cnt_smi[smi] += 1
                    self.smi2macro_ids[smi].append((macro_node_id, nei_macro_node))
        # return cnt_smi, self.smi2macro_ids
        return cnt_smi
    
    def merge(self, smi, tokenizer=None):
        # merge two nodes if merged nodes = smi
        
        if smi not in self.smi2macro_ids:
            return 0
        for macro_node_id1, macro_node_id2 in self.smi2macro_ids[smi]:
            if tokenizer is not None:
                fragment_idx = tokenizer.smi_id(smi)
                self.fragment_idx_dict[self.macro_nodes_id] = self.fragment_idx_dict[macro_node_id1] + self.fragment_idx_dict[macro_node_id2] + [fragment_idx]
            if macro_node_id1 not in self.macro_nodes_dict or macro_node_id2 not in self.macro_nodes_dict:
                continue
            atom_idxs = self.macro_nodes_dict[macro_node_id1][1] | self.macro_nodes_dict[macro_node_id2][1]
            node_idxs = self.macro_nodes_dict[macro_node_id1][0] + self.macro_nodes_dict[macro_node_id2][0]
            new_macro_nodes = (node_idxs, atom_idxs)
            self.macro_nodes_dict[self.macro_nodes_id] = new_macro_nodes
            self.macro_nodes_dict.pop(macro_node_id1)
            self.macro_nodes_dict.pop(macro_node_id2)
            for node_id in node_idxs:
                # update node to macro node mapping
                self.nodeID_to_macroID[node_id] = self.macro_nodes_id
            self.macro_nodes_id += 1
        return 1

def freq_cnt_new(mol):
    return mol.get_nei_smis(), mol

def graph_dove(fname, vocab_len, vocab_path, order=1, cpus=16, kekulize=False):
    # order: L(G)的阶数，即迭代次数
    # load molecules
    print_log(f'Loading mols from {fname} ...')
    with open(fname, 'r') as fin:
        smis = list(map(lambda x: x.strip(), fin.readlines()))
    # init to atoms
    # smis = smis[:10000]
    mols = []
    global_cnt_smi = defaultdict(int)
    
    for smi in tqdm(smis):
        rdkit_mol = smi2mol(smi, kekulize)
        mol = MolGraph(rdkit_mol)
        for _ in range(order):
            mol.line_graph_transform()
            if order>=3:
                mol.remove_duplicate_nodes()
        mol.build_init_macro_nodes()
        mol.build_adj_table()
        mols.append(mol)
        cnt_smi = mol.get_nodes_smiles()
        # print(cnt_smi);exit()
        for smi in cnt_smi:
            global_cnt_smi[smi] += cnt_smi[smi]
    add_len = vocab_len
    selected_smis = []
    print_log(f'Added {len(global_cnt_smi)} init fragments, {add_len} principal subgraphs to extract')
    pbar = tqdm(total=add_len)
    pool = mp.Pool(cpus)
    while len(selected_smis) < vocab_len:
        # res_list = [mol.get_nei_smis() for mol in mols]
        res_list = list(pool.map(freq_cnt_new, mols))  # each element is (freq, mol) (because mol will not be synced...)
        # print(res_list[0])
        res_list, mols = list(zip(*res_list))
        # for res, mol in zip(res_list, mols):
        #     mol.smi2macro_ids = res[1] 
            # 注意多线程时，会用不同线程处理mol，在另一个线程中更改mol，不会反映到主进程中，所以需要在函数中返回mol!
        # res_list = [res[0] for res in res_list]
        # exit()
        freqs = defaultdict(int)
        for freq in res_list:
            for key in freq:
                freqs[key] += freq[key]
        # find the subgraph to merge
        # 代码可以简化成一行
        # print(freqs)
        merge_smi, max_cnt = max(freqs.items(), key=lambda x: x[1])
        # merge
        for mol in mols:
            mol.merge(merge_smi)
        if merge_smi in global_cnt_smi:  # corner case: re-extracted from another path
            continue
        selected_smis.append(merge_smi)
        # num_atoms = cnt_atom(merge_smi)

        global_cnt_smi[merge_smi] = max_cnt
        print("merge_smi", merge_smi)
        pbar.update(1)
        # print("mol", mols[0])
    pbar.close()
    print_log('sorting vocab by atom num')
    global_cnt_smi = {smi: [cnt_atom(smi), freq] for smi, freq in global_cnt_smi.items()}
    selected_smis.sort(key=lambda x: global_cnt_smi[x][0], reverse=True)
    pool.close()
    print(f"vocab size: {len(global_cnt_smi)}")
    with open(vocab_path, 'w') as fout:
        fout.write(json.dumps({'kekulize': kekulize}) + '\n')
        fout.writelines(list(map(lambda smi: f'{smi}\t{global_cnt_smi[smi][0]}\t{global_cnt_smi[smi][1]}\n', global_cnt_smi.keys())))
    return selected_smis, global_cnt_smi

class TokenizerNew:
    def __init__(self, vocab_path, order=0):
        self.order = order
        with open(vocab_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        # load kekulize config
        config = json.loads(lines[0])
        self.kekulize = config['kekulize']
        lines = lines[1:]
        
        self.vocab_dict = {}
        self.idx2subgraph, self.subgraph2idx = [], {}
        self.max_num_nodes = 0
        for line in lines:
            smi, atom_num, freq = line.strip().split('\t')
            self.vocab_dict[smi] = (int(atom_num), int(freq))
            self.max_num_nodes = max(self.max_num_nodes, int(atom_num))
            self.subgraph2idx[smi] = len(self.idx2subgraph)
            self.idx2subgraph.append(smi)
        # self.pad, self.end = '<pad>', '<s>'
        # for smi in [self.pad, self.end]:
        #     self.subgraph2idx[smi] = len(self.idx2subgraph)
        #     self.idx2subgraph.append(smi)
        # # for fine-grained level (atom level)
        # self.bond_start = '<bstart>'
        # self.max_num_nodes += 2 # start, padding
        unk_smi = '<unk>' # 未知的子图
        self.subgraph2idx[unk_smi] = len(self.idx2subgraph)
        self.idx2subgraph.append(unk_smi)
        self.max_num_nodes += 1
        
    def smi_id(self, smi):
        if smi in self.subgraph2idx:
            return self.subgraph2idx[smi]
        else:
            return self.subgraph2idx['<unk>']
        
    def tokenize(self, mol):
        # smiles = mol
        # if isinstance(mol, str):
        #     mol = smi2mol(mol, self.kekulize)
        # else:
        #     smiles = mol2smi(mol)
        # rdkit_mol = mol
        # mol = MolGraph(mol)
        for _ in range(self.order):
            mol.line_graph_transform()
            if self.order>=3:
                mol.remove_duplicate_nodes()
        mol.build_init_macro_nodes()
        mol.build_adj_table()
        mol.init_macro_nodes_fragment_idx(self)
        while True and self.order<=2:
            nei_smis = mol.get_nei_smis()
            max_freq, merge_smi = -1, ''
            # 找到"频次"最多的子图，并merge，直到无法merge
            # "频次"指的是字典中子图的频次
            for smi in nei_smis:
                if smi not in self.vocab_dict:
                    continue
                freq = self.vocab_dict[smi][1]
                if freq > max_freq:
                    max_freq, merge_smi = freq, smi
            if max_freq == -1:
                break
            mol.merge(merge_smi, self)
        # res = mol.get_smis_subgraphs() # a list of (smi, idxs) which corresponds to a subgraph
        res = mol.get_smis_subgraphs_new()
        # construct reversed index
        aid2pid = {}
        for pid, subgraph in enumerate(res): # 重新映射atom idx到pid
            _, aids, _ = subgraph
            for aid in aids:
                aid2pid[aid] = pid
        group_idxs = [x[1] for x in res] # 
        fragment_smiless = [x[0] for x in res]
        group_idxs_detail = [x[2] for x in res]
        # fragment_idxs = [self.subgraph2idx[s] for s in fragment_smiless] 
        fragment_idxs = [self.smi_id(s) for s in fragment_smiless] 
        
        return group_idxs, fragment_idxs, group_idxs_detail
        # return group_idxs, fragment_smiless, fragment_idxs
        # return rdkit_mol, group_idxs, fragment_smiless, fragment_idxs
        # return Molecule(rdkit_mol, group_idxs, self.kekulize)

    def idx_to_subgraph(self, idx):
        return self.idx2subgraph[idx]
    
    def subgraph_to_idx(self, subgraph):
        return self.subgraph2idx[subgraph]
    
    def pad_idx(self):
        return self.subgraph2idx[self.pad]
    
    def end_idx(self):
        return self.subgraph2idx[self.end]
    
    def atom_vocab(self):
        return copy(self.atom_level_vocab)

    def num_subgraph_type(self):
        return len(self.idx2subgraph)
    
    def atom_pos_pad_idx(self):
        return self.max_num_nodes - 1
    
    def atom_pos_start_idx(self):
        return self.max_num_nodes - 2

    def __call__(self, mol):
        return self.tokenize(mol)
    
    def __len__(self):
        return len(self.idx2subgraph)


def parse():
    parser = argparse.ArgumentParser(description='dove-k')
    parser.add_argument('--smiles', type=str, default='COc1cc(C=NNC(=O)c2ccc(O)cc2O)ccc1OCc1ccc(Cl)cc1',
                        help='The molecule to tokenize (example)')
    parser.add_argument('--data', type=str, required=True, help='Path to molecule corpus')
    parser.add_argument('--vocab_size', type=int, default=500, help='Length of vocab')
    parser.add_argument('--output', type=str, required=True, help='Path to save vocab')
    parser.add_argument('--workers', type=int, default=16, help='Number of cpus to use')
    parser.add_argument('--kekulize', action='store_true', help='Whether to kekulize the molecules (i.e. replace aromatic bonds with alternating single and double bonds)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    graph_dove(args.data, vocab_len=args.vocab_size, vocab_path=args.output,
              cpus=args.workers, kekulize=args.kekulize)
