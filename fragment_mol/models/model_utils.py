# import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import copy 

class MLP(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_layers, activation, d_hidden_feats, dropout=0.0):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        if self.n_layers == 1:
            self.linear = nn.Linear(d_in_feats, d_out_feats)
        else:
            self.d_hidden_feats = d_hidden_feats
            self.layer_list = nn.ModuleList()
            self.in_proj = nn.Linear(d_in_feats, self.d_hidden_feats)
            for _ in range(self.n_layers-2):
                self.layer_list.append(nn.Linear(self.d_hidden_feats, self.d_hidden_feats))
            # print(self.d_hidden_feats, d_out_feats)
            self.out_proj = nn.Linear(self.d_hidden_feats, d_out_feats)
            self.act = activation
            self.hidden_dropout = nn.Dropout(dropout)
        
    def forward(self, feats):
        if self.n_layers == 1:
            return self.linear(feats)
        else:
            feats = self.act(self.in_proj(feats))
            feats = self.hidden_dropout(feats)
            for i in range(self.n_layers-2):
                feats = self.act(self.layer_list[i](feats))
                feats = self.hidden_dropout(feats)
            feats = self.out_proj(feats)
            return feats
    
def get_predictor(d_input_feats, n_tasks, n_layers, d_hidden_feats, predictor_drop=0.0):
    return MLP(d_input_feats, n_tasks, n_layers, nn.GELU(), d_hidden_feats, predictor_drop)

def model_param_norm(model):
    param_norm = 0
    for param in model.parameters():
        param_norm += torch.norm(param.data, 2) ** 2
    param_norm = param_norm.sqrt()
    return param_norm.item()

def model_grad_norm(model):
    grad_norm = 0
    for param in model.parameters():
        grad_norm += torch.norm(param.grad, 2) ** 2
    grad_norm = grad_norm.sqrt()
    return grad_norm.item()

def model_grad_norm_ratio(model):
    ratio_list = []
    for param in model.parameters():
        if len(param.grad.shape) == 1: 
            # ignore bias term
            continue
        ratio = torch.norm(param.grad, 2) / torch.norm(param.data, 2)
        ratio_list.append(ratio.item())
    return ratio_list

def model_n_params(model):
    return sum([p.numel() for p in model.parameters()])

def module_loss(model):
    m_loss = 0
    for name, m in model.named_modules():
        if hasattr(m,'loss'):
            #print(name)
            m_loss += m.loss()
    return m_loss

def random_lp_flip(pe):
    flip = 2*(torch.rand(pe.shape[1]) > 0.5)-1
    return flip*pe 

def random_svd_flip(pe, k):
    flip = 2*(torch.rand(k) > 0.5)-1
    multiplier = torch.cat([flip, flip, torch.ones(pe.shape[1]-2*k)])
    return multiplier*pe 


def positional_encoding(graph, target_dim, pe_type='random_walk', random_flip=True):
    if pe_type == 'random_walk':
        return dgl.random_walk_pe(graph, k=target_dim)
    elif pe_type == 'laplace':
        pe = dgl.lap_pe(graph, k=target_dim, padding=True)
        if random_flip:
            pe = random_lp_flip(pe)
    elif pe_type == 'svd':
        
        pe = dgl.svd_pe(graph, k=target_dim//2, padding=True, random_flip=False)
        if random_flip:
            pe = random_svd_flip(pe, min(target_dim//2, graph.num_nodes()))
       
    elif pe_type == 'dummy':
        pe = torch.zeros(graph.num_nodes(), target_dim)
    else:
        raise ValueError('Unknown positional encoding type')
    return pe

class ModelWithEMA(nn.Module):
    def __init__(self, model, decay=0.9):
        super().__init__()
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

    def update_ema(self, new_model):
        ema_model_dict = self.ema_model.state_dict()
        model_dict = new_model.state_dict()
        for k, v in model_dict.items():
            if k in ema_model_dict:
                ema_model_dict[k].data.copy_(
                    self.decay * ema_model_dict[k].data + (1.0 - self.decay) * v.data)
        self.ema_model.load_state_dict(ema_model_dict)

    
