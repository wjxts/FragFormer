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
from fragment_mol.models.model_gnn import ProjIn

__all__ = ["GraphTransformer"]

class Residual(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_ffn_dense_layers, feat_drop, activation):
        super(Residual, self).__init__()
        self.norm = nn.LayerNorm(d_in_feats)
        self.in_proj = nn.Linear(d_in_feats, d_out_feats)
        self.ffn = MLP(d_out_feats, d_out_feats, n_ffn_dense_layers, activation, d_hidden_feats=d_out_feats*4)
        self.feat_dropout = nn.Dropout(feat_drop)

    def forward(self, x, y):
        x = x + self.feat_dropout(self.in_proj(y))
        y = self.norm(x)
        y = self.ffn(y)
        y = self.feat_dropout(y)
        x = x + y
        return x


class GraphTransformerLayer(nn.Module):
    def __init__(self,
                d_feats,
                n_heads,
                n_ffn_dense_layers,
                feat_drop=0.,
                attn_drop=0.,
                activation=nn.GELU()):
        super().__init__()
        self.d_feats = d_feats
        # self.d_trip_path = d_feats//d_hpath_ratio
        # self.path_length = path_length
        self.n_heads = n_heads
        self.scale = d_feats**(-0.5)
        
        
        self.attention_norm = nn.LayerNorm(d_feats)
        self.qkv = nn.Linear(d_feats, d_feats*3)
        self.node_out_layer = Residual(d_feats, d_feats, n_ffn_dense_layers, feat_drop,  activation)
        
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.act = activation
       

    def pretrans_edges(self, edges):
        edge_h = edges.src['hv']
        return {"he": edge_h}

    def forward(self, g, node_feature):
        g = g.local_var() 
        
        dist_attn = g.edata['dist_attn']
        path_attn = g.edata['path_attn']
        node_feature = self.attention_norm(node_feature)
        qkv = self.qkv(node_feature).reshape(-1, 3, self.n_heads, self.d_feats // self.n_heads).permute(1, 0, 2, 3)
        q, k, v = qkv[0]*self.scale, qkv[1], qkv[2]
    
        g.dstdata.update({'K': k})
        g.srcdata.update({'Q': q})
        g.apply_edges(fn.u_dot_v('Q', 'K', 'node_attn'))
        
        g.edata['a'] = g.edata['node_attn'] + dist_attn.reshape(len(g.edata['node_attn']), -1, 1) + path_attn.reshape(len(g.edata['node_attn']), -1, 1) 
        g.edata['sa'] = self.attn_dropout(edge_softmax(g, g.edata['a']))

        g.ndata['hv'] = v.view(-1, self.d_feats)
        g.apply_edges(self.pretrans_edges)
        g.edata['he'] = ((g.edata['he'].view(-1, self.n_heads, self.d_feats//self.n_heads)) * g.edata['sa']).view(-1, self.d_feats)
        
        g.update_all(fn.copy_e('he', 'm'), fn.sum('m', 'agg_h'))
        return self.node_out_layer(node_feature, g.ndata['agg_h'])

class PathAttentionScore(nn.Module):
    def __init__(self, hidden_size=128, max_length=5, head=1) -> None:
        super().__init__()
        self.max_length = max_length
        self.head = head
       
        self.trip_fortrans = nn.ModuleList([
          nn.Linear(hidden_size, 1, bias=False) for _ in range(max_length+1)
        ])
        
    def forward(self, paths, node_feature):
       
        paths[paths<0] = -1
       
        attn_scores = []

        for i in range(self.max_length+1):
            idxs = paths[:, i]
            s = torch.cat([self.trip_fortrans[i](node_feature), torch.zeros(size=(1, self.head)).to(node_feature)], dim=0)[idxs]
            attn_scores.append(s)
        path_length = torch.sum(paths>=0, dim=-1, keepdim=True).clip(min=1)
        attn_score = torch.sum(torch.stack(attn_scores, dim=-1), dim=-1)
        attn_score = attn_score/path_length
        return attn_score 

@register_model_arg_func("graph_transformer")
def graph_transformer_model_args():
    parser = argparse.ArgumentParser(description="GNN model args")
    parser.add_argument('--input_form', type=str, default="graph", help='input form',
                        choices=['line_graph', 'graph'])
    parser.add_argument('--in_feats', type=int, default=ATOM_FEATURE_DIM, help='input feature dimension')
    parser.add_argument('--d_g_feats', type=int, default=128, help='model dimension')
    parser.add_argument('--n_mol_layers', type=int, default=2, help='# of layers')
    parser.add_argument('--n_ffn_dense_layers', type=int, default=2, help='# of dense layers in FFN')
    parser.add_argument('--n_heads', type=int, default=2, help='# of heads in MSA')
    parser.add_argument('--readout', type=str, default='sum', help='readout function')
    parser.add_argument('--feat_drop', type=float, default=0.0, help='feature dropout')
    parser.add_argument('--attn_drop', type=float, default=0.0, help='attention dropout')
    
    args, _ = parser.parse_known_args()
    return args 

@register_model("graph_transformer")
class GraphTransformer(nn.Module):
    def __init__(self,
                args,
                n_tasks,
                ):
        super().__init__()
        self.n_mol_layers = args.n_mol_layers
        self.n_heads = args.n_heads
        activation=nn.GELU()
        self.distance_embedding = nn.Embedding(100, 1)
        # initialize distance_embedding to  0 
        nn.init.constant_(self.distance_embedding.weight, 0)
        self.path_embedding = PathAttentionScore(args.d_g_feats, max_length=5)
        self.centality_embedding = nn.Embedding(100, args.d_g_feats)
        self.d_g_feats = args.d_g_feats
        # self.proj_in = nn.Linear(args.in_feats, args.d_g_feats)
        self.proj_in = ProjIn(args.input_form, args.in_feats, args.d_g_feats)
        # Molecule Transformer Layers
        self.mol_T_layers = nn.ModuleList([
            GraphTransformerLayer(args.d_g_feats, args.n_heads, args.n_ffn_dense_layers, args.feat_drop, args.attn_drop, activation) for _ in range(args.n_mol_layers)
        ])
        self.readout = args.readout
        self.predictor = get_predictor(args.d_g_feats, n_tasks, 2, args.d_g_feats, 0.1)
        
    
    def forward(self, data):
        g = data['g']
        
        
        node_feature = self.proj_in(g)
   
        distance = g.edata['distance']
        g.edata['dist_attn'] = self.distance_embedding(distance)
        paths = g.edata['paths']
        g.edata['path_attn'] = self.path_embedding(paths, node_feature)
        for i in range(self.n_mol_layers):
            node_feature = self.mol_T_layers[i](g, node_feature)
        g.ndata['h'] = node_feature
        x = dgl.readout_nodes(g, 'h', op=self.readout)
      
        x = self.predictor(x)
        return x