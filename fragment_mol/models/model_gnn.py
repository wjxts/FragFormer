import dgl
from dgl.nn.functional import edge_softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from fragment_mol.models.model_utils import get_predictor, MLP
import argparse

from fragment_mol.utils.chem_utils import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from fragment_mol.utils.fingerprint import FP_FUNC_DICT, FP_DIM
from fragment_mol.register import register_model, register_model_arg_func
from fragment_mol.modules.observer import Observer


GNN_LAYERS = {}
def register_gnn_layer(name):
    def decorator(gnn_layer):
        GNN_LAYERS[name] = gnn_layer
        return gnn_layer
    return decorator 


@register_gnn_layer('gcn')
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_feats))
        self.act = nn.ReLU()
        self.observer = Observer()
        
    
    def forward(self, g, feature):
        feature = self.observer(feature)
        with g.local_scope():
            degrees = g.in_degrees().float().clamp(min=1).unsqueeze(-1)
            g.ndata['h'] = self.linear(feature/torch.sqrt(degrees))
            g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'h'))
            h = g.ndata['h']/torch.sqrt(degrees)
            h = h + self.bias
            h = self.act(h)
            return h

@register_gnn_layer('gin')
class GINLayer(nn.Module):
    def __init__(self, in_feats, out_feats, learn_eps=True, init_eps=0.) -> None:
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        if learn_eps:
            self.register_parameter('eps', nn.Parameter(torch.Tensor([init_eps])))
        else:
            self.register_buffer('eps', torch.Tensor([init_eps]))
        self.act = nn.ReLU()
    
    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'h'))
            h = self.linear((1+self.eps)*g.ndata['h'])
            h = self.act(h)
            return h
        
@register_gnn_layer('gat')
class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads=2):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        self.a = nn.Parameter(torch.zeros(num_heads, out_feats*2//num_heads))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)
        self.attn_linear = nn.Linear(in_feats, out_feats, bias=False)
        self.num_heads = num_heads
        self.out_feats = out_feats
        
    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            def cal_score_func(edges):
                src = edges.src 
                dst = edges.dst
               
                src_key = self.attn_linear(src['h']).reshape(-1, self.num_heads, self.out_feats//self.num_heads)
                dst_key = self.attn_linear(dst['h']).reshape(-1, self.num_heads, self.out_feats//self.num_heads)
                key = torch.cat([src_key, dst_key], dim=-1)
                score = (key * self.a.unsqueeze(0)).sum(-1)
                score = self.attn_act(score)
                return {'score': score}

            g.apply_edges(cal_score_func)
            g.edata['a'] = edge_softmax(g, g.edata['score'])
            def message_func(edges):
                src = edges.src 
                dst = edges.dst

                message = g.edata['a'].unsqueeze(-1) * self.linear(src['h']).reshape(-1, self.num_heads, self.out_feats//self.num_heads)
                message = message.reshape(-1, self.out_feats)
                return {'m': message}

            def reduce_func(nodes):
                
                return {'newh': nodes.mailbox['m'].sum(1)}
            g.update_all(message_func, reduce_func)
            h = g.ndata['newh']
            return h


 
class LineGraphProjIn(nn.Module):
    def __init__(self, d_g_feats):
        super().__init__()
        d_atom_feats = ATOM_FEATURE_DIM
        d_edge_feats = BOND_FEATURE_DIM
        
        self.in_proj_atom = nn.Linear(d_atom_feats, d_g_feats, bias=False)
        self.in_proj_edge = nn.Linear(d_edge_feats, d_g_feats, bias=False)
        
        self.in_proj_triple = nn.Linear(d_g_feats*2, d_g_feats, bias=False)
        
    def forward(self, data):
        begin_end_feature, edge_feature = data
        atom_feature = self.in_proj_atom(begin_end_feature).sum(dim=1)
        edge_feature = self.in_proj_edge(edge_feature)
        triplet_feature = torch.cat([atom_feature, edge_feature], dim=-1)
        
        return self.in_proj_triple(triplet_feature)
        
class ProjKnode(nn.Module):
    def __init__(self, d_g_feats):
        super().__init__()
        d_fp_feats = 512
        d_md_feats = 200
        self.in_proj_fp = MLP(d_fp_feats, d_g_feats, 2, nn.GELU())
        self.in_proj_md = MLP(d_md_feats, d_g_feats, 2, nn.GELU())
        
        
    def forward(self, triplet_feature, indicators, fp, md):
        triplet_feature[indicators==1] = self.in_proj_fp(fp)
        triplet_feature[indicators==2] = self.in_proj_md(md)
        return triplet_feature

class ProjIn(nn.Module):
    def __init__(self, d_in, d_out, line_g: bool=False) -> None:
        super().__init__() 
        self.line_g = line_g
        if not line_g:
            self.proj_in = nn.Linear(d_in, d_out, bias=False)
        else:
            self.proj_in = LineGraphProjIn(d_g_feats=d_out)
        
    def forward(self, g):
        if not self.line_g:
            x = g.ndata['h']
        else:
            x = (g.ndata['begin_end'], g.ndata['edge'])
        x = self.proj_in(x)
        return x

class KnowledgeFusion(nn.Module):
    def __init__(self, d_model, fp_name='ecfp'):
        super().__init__()
        self.d_model = d_model
        self.proj_knowledge = nn.Linear(FP_DIM[fp_name], d_model, bias=False)
        self.gru = nn.GRUCell(d_model, d_model)
        
    def forward(self, bg, node_feature, k_feature):
        k_feature = self.proj_knowledge(k_feature)
        k_feature = dgl.broadcast_nodes(bg, k_feature)
        node_feature = self.gru(k_feature, node_feature)
        return node_feature
        
class KnowledgePooling(nn.Module):
    def __init__(self, d_model, fp_name='ecfp', readout='sum'):
        super().__init__()
        # 可以变为multi-head
        self.d_model = d_model
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_q = nn.Linear(FP_DIM[fp_name], d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model)
        self.readout = readout
        self.gru = nn.GRUCell(d_model, d_model)
        
    def forward(self, bg, node_feature, k_feature):
        with bg.local_scope():
            k_feature = self.linear_q(k_feature)

            q = dgl.broadcast_nodes(bg, k_feature)
            k = self.linear_k(node_feature/self.d_model**0.5)
            v = self.linear_v(node_feature)

            score = (k*q).sum(-1)
            bg.ndata['score'] = score
            bg.ndata['attn'] = dgl.softmax_nodes(bg, 'score')
            bg.ndata['h'] = bg.ndata['attn'].unsqueeze(-1)*v
            out = dgl.readout_nodes(bg, 'h', op=self.readout) 
         
            return out + k_feature
            
        
        
@register_model_arg_func('gnn')
def gnn_model_args():
    parser = argparse.ArgumentParser(description="GNN model args")
    parser.add_argument('--input_form', type=str, default="graph", help='input form',
                        choices=['line_graph', 'graph', 'frag_graph', 'frag_graph_gnn'])
    parser.add_argument('--line_g', action='store_true', help='whether to use line graph')
    parser.add_argument('--complete_g', action='store_true', 
                        help='whether to use complete graph, for graph transformer')
    parser.add_argument('--layer_name', type=str, default='gcn', help='layer type', choices=list(GNN_LAYERS.keys()))
    parser.add_argument('--in_feats', type=int, default=ATOM_FEATURE_DIM, help='input feature dimension')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
    parser.add_argument('--layer_num', type=int, default=3, help='# of layers')
    parser.add_argument('--subgraph_layer_num', type=int, default=2, help='# of subgraph layers')
    parser.add_argument('--frag_layer_num', type=int, default=1, help='# of fragment graph layers')
    parser.add_argument('--pe_dim', type=int, default=6, help='dim of positional encoding')
    parser.add_argument('--readout', type=str, default='sum', help='readout function')
    parser.add_argument("--knodes", type=str, default=[], nargs="*", help="knowledge type",
                        choices=list(FP_FUNC_DICT.keys()))
    args, _ = parser.parse_known_args()
    return args 

@register_model('gnn')
class GNN(nn.Module):
    def __init__(self, args, n_tasks):
        super().__init__()
        print(args)
        self.knodes = args.knodes 
        self.input_form = args.input_form
        # print(f"model input form: {args.input_form}")
        print(f"line_g: {args.line_g}, complete_g: {args.complete_g}")
        self.proj_in = ProjIn(args.in_feats, args.hidden_size, args.line_g)
        self.proj_pe = nn.Linear(args.pe_dim, args.hidden_size, bias=False)
        layer_cls = GNN_LAYERS[args.layer_name]
        self.gnns = nn.ModuleList([layer_cls(args.hidden_size, args.hidden_size) for _ in range(args.layer_num)])
        # 这里readout用sum好一些
        self.k_pools = nn.ModuleList([KnowledgePooling(args.hidden_size, fp_name=fp_name, readout='mean') for fp_name in self.knodes])
       
        self.readout = 'mean'

    
        self.weight = nn.Parameter(torch.zeros((len(args.knodes)+1)))
        self.predictor = get_predictor(args.hidden_size, n_tasks, 1, args.hidden_size, 0.1) 
 
        
        
    def forward(self, data):
        g = data['g']
        x = self.proj_in(g)

        for gnn in self.gnns:
        g.ndata['h'] = x
        self.gap_activations = x
        x = dgl.readout_nodes(g, 'h', op=self.readout)
        x = self.predictor(x)
        return x
    
    def store_gap_activation_grap(self, grad):
        self.gap_activations_grad = grad
    
    def get_gap_activations(self):
        return self.gap_activations.detach()
    
    def get_prediction_weights(self):
        if not hasattr(self.predictor, 'linear'):
            print("no gap layer!")
            exit()
        return self.predictor.linear.weight
    
    def get_gap_gradients(self):
        return self.gap_activations_grad.detach()

def test():
    args = gnn_model_args()
    g = dgl.graph(([0,1,2,3,4,5,6], [1,2,3,4,5,6,7]))
    g.edata['e'] = torch.randn((7, 5))
    g = dgl.add_self_loop(g)
    
    # 指定节点特征并且初始化模型
    features = torch.randn((8, 16))
    g = dgl.batch([g, g])
    features = torch.cat([features, features], dim=0)
    g.ndata['h'] = features
    net = GNN(args, n_tasks=10)
    
   
    logits = net({"g": g})
    print(logits.shape)

