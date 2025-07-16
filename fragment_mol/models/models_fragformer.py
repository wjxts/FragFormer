import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.functional import edge_softmax
import numpy as np
from fragment_mol.models.model_utils import get_predictor, MLP
from fragment_mol.register import register_model, register_model_arg_func
import argparse
from fragment_mol.utils.chem_utils import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from fragment_mol.utils.fingerprint import FP_FUNC_DICT
from fragment_mol.models.model_gnn import ProjIn, GINLayer, KnowledgePooling, KnowledgeFusion
from fragment_mol.models.model_graph_transformer import GraphTransformerLayer, PathAttentionScore
import torch_scatter 
import math 

from vector_quantize_pytorch import VectorQuantize

__all__ = ["FragFormer"]


@register_model_arg_func("fragformer")
def graph_transformer_model_args():
    parser = argparse.ArgumentParser(description="GNN model args")
    parser.add_argument('--input_form', type=str, default="graph", help='input form',
                        choices=['line_graph', 'graph', 'frag_graph', 
                                 'jt_graph', ])
    parser.add_argument('--in_feats', type=int, default=ATOM_FEATURE_DIM, help='input feature dimension')
    parser.add_argument('--edge_feats', type=int, default=BOND_FEATURE_DIM, help='input edge feature dimension')
    parser.add_argument('--d_model', type=int, default=128, help='model dimension')
    parser.add_argument('--n_subg_layers', type=int, default=2, help='# of subgraph encoder layers')
    parser.add_argument('--n_mol_layers', type=int, default=2, help='# of fragformer layers')
    parser.add_argument('--n_ffn_dense_layers', type=int, default=2, help='# of dense layers in FFN')
    parser.add_argument('--n_heads', type=int, default=2, help='# of heads in MSA')
    parser.add_argument('--readout', type=str, default='sum', help='readout function')
    parser.add_argument('--feat_drop', type=float, default=0.0, help='feature dropout')
    parser.add_argument('--attn_drop', type=float, default=0.0, help='attention dropout')
    parser.add_argument("--knodes", type=str, default=[], nargs="*", help="knowledge type",
                        choices=list(FP_FUNC_DICT.keys()))
    parser.add_argument('--vq', action='store_true', help='whether use vq')
    args, _ = parser.parse_known_args()
    return args 



class SubgraphPooling(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, node_feature, batch_node_ids, batch_macro_node_ids):
        fragment_subgraph_embed = torch_scatter.scatter(node_feature[batch_node_ids], batch_macro_node_ids, dim=0, reduce='mean')
        return fragment_subgraph_embed

class GraphPooling(nn.Module):
    def __init__(self, readout='sum') -> None:
        super().__init__() 
        self.readout = readout
    
    def forward(self, node_feature, g):
        g.ndata['_pool_h'] = node_feature
        x = dgl.readout_nodes(g, '_pool_h', op=self.readout)
        return x
    
@register_model("fragformer")
class FragFormer(nn.Module):
    def __init__(self, args, n_tasks):
        super().__init__()
        self.knodes = args.knodes
        self.d_model = args.d_model
        self.use_vq = args.vq
        self.node_proj_in = ProjIn(args.in_feats, args.d_model)
        
        self.edge_proj_in = nn.Linear(args.edge_feats, args.d_model, bias=False)
        self.subgraph_encoder = nn.ModuleList([GINLayer(args.d_model, args.d_model) for _ in range(args.n_subg_layers)])
        self.subgraph_pooling = SubgraphPooling()
        self.fragment_transformer = nn.ModuleList([
            GraphTransformerLayer(args.d_model, args.n_heads, args.n_ffn_dense_layers, args.feat_drop, args.attn_drop, nn.GELU()) for _ in range(args.n_mol_layers)
        ])
        if self.use_vq:
            self.vq = VectorQuantize(
                    dim = args.d_model,      # input dim
                    codebook_size = 256,     # codebook size
                    decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
                    commitment_weight = 1.   # the weight on the commitment loss
                    )
        self.distance_embedding = nn.Embedding(100, 1)
        # initialize distance_embedding to 0 
        nn.init.constant_(self.distance_embedding.weight, 0)
        self.path_embedding = PathAttentionScore(args.d_model, max_length=2)
        
        self.fragment_graph_pooling = GraphPooling('sum')
        self.k_pools = nn.ModuleList([KnowledgePooling(args.d_model, fp_name=fp_name, readout='mean') for fp_name in self.knodes])
        self.k_fusions = nn.ModuleList([KnowledgeFusion(args.d_model, fp_name=fp_name) for fp_name in self.knodes])
        self.weight = nn.Parameter(torch.zeros((len(args.knodes)+1)))
        
        self.predictor = get_predictor(args.d_model, n_tasks, 2, args.d_model, 0.1)
    
    def init_ft_predictor(self, n_tasks, dropout=0.1):
        del self.predictor
        self.predictor = get_predictor(self.d_model, n_tasks, 1, self.d_model, 0)
    
    def loss(self):
        return self.vq_loss 
    
    def forward(self, data, pretrain=False, max_idx=None, lift=None):
        
        vanilla_g = data['vanilla_g']
        fragformer_g = data['fragformer_g']
        node_feature = self.node_proj_in(vanilla_g)
       
        for layer in self.subgraph_encoder:
            node_feature = layer(vanilla_g, node_feature)
        fragment_subgraph_embed = self.subgraph_pooling(node_feature, data['batch_node_ids'], data['batch_macro_node_ids'])
        
       
        if self.use_vq:
            quantized, indices, commit_loss = self.vq(fragment_subgraph_embed) 
            
            self.vq_loss = commit_loss[0]
            fragment_subgraph_embed = fragment_subgraph_embed + quantized
        else:
            self.vq_loss = 0
        
       
        fragment_feature = fragment_subgraph_embed 
        
        distance = fragformer_g.edata['distance']
        paths = fragformer_g.edata['paths'] 
        fragformer_g.edata['dist_attn'] = self.distance_embedding(distance)
     
        fragformer_g.edata['path_attn'] = self.path_embedding(paths, fragment_feature)
        
        # mask fragment 
        mask = data['fragment_mask']
        fragment_feature = fragment_feature*(1-mask.to(fragment_feature)[:, None])
        
        for i, fp_name in enumerate(self.knodes):

            k_feature = data['knodes'][fp_name] 
            fragment_feature = self.k_fusions[i](fragformer_g, fragment_feature, k_feature)
        
        for layer in self.fragment_transformer:
           
            fragment_feature = (layer(fragformer_g, fragment_feature) + fragment_feature)
        
       
        if pretrain:
            logits = self.predictor(fragment_feature)
 
            return logits
        
        pool_feature_list = []
        for i, fp_name in enumerate(self.knodes):
            k_feature = data['knodes'][fp_name]
            pool_feature_list.append(self.k_pools[i](fragformer_g, fragment_feature, k_feature))
        
        if max_idx is not None:
            fragment_feature[max_idx] = 0
        
        
        x = self.fragment_graph_pooling(fragment_feature, fragformer_g)
        
        if len(pool_feature_list) > 0:
            weighted_feature = torch.stack([x]+pool_feature_list, dim=-1)*F.softmax(self.weight, dim=0)
            x = weighted_feature.sum(-1)
        
        logits = self.predictor(x)

        return logits
