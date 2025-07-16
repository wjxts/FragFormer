import torch 
import torch.nn as nn 

import dgl 

from fragment_mol.utils.fingerprint import FP_DIM

from typing import List

from fragment_mol.models.models_fragformer import KnowledgeFusion

class AtttentiveFusion(nn.Module):
   
    def __init__(self, d_model: int, knodes: List[str]):
        super().__init__()
        self.d_model = d_model
        self.knodes = knodes
        self.d_attn = d_model // 4
        self.k_proj = nn.ModuleDict([
            (k, nn.Linear(FP_DIM[k], self.d_attn, bias=False)) for k in knodes
        ])
        
        self.v_proj = nn.ModuleDict([
            (k, nn.Linear(FP_DIM[k], d_model)) for k in knodes
        ])
        
        self.Wq = nn.Linear(d_model, self.d_attn, bias=False)
      
        self.steps = 0

    
    def build_vectors(self, bg, knodes, proj):
        # knodes: a dict of knowledge nodes
        vectors = []
        for fp_name, v in knodes.items():
            feature = proj[fp_name](v) # bs*d
            feature = dgl.broadcast_nodes(bg, feature) # N*d
            vectors.append(feature)
        vectors = torch.stack(vectors, dim=1) # N*k*d
        return vectors
    
    def forward(self, bg, x, knodes):
        # knodes: a dict of knowledge nodes {fp_name: bs*fp_dim}
        # x: N*d
        if len(knodes) == 0:
            return x
        k_vectors = self.build_vectors(bg, knodes, self.k_proj)  # N*k*d_attn
        v_vectors = self.build_vectors(bg, knodes, self.v_proj)  # N*k*d
       
        q = self.Wq(x) / (self.d_model**0.5) # N*d_attn
        q = q.unsqueeze(1) # N*1*d_attn
        
        attn = torch.bmm(q, k_vectors.transpose(1, 2)) # N*1*k

        attn = torch.softmax(attn, dim=-1) # N*1*k
        
        out = torch.bmm(attn, v_vectors).squeeze(1) # N*d
        
        return x + out


class GRUFusion(nn.Module):
    def __init__(self, d_model: int, knodes: List[str]):
        super().__init__()
        self.d_model = d_model
        self.knodes = knodes
        self.k_fusions = nn.ModuleList([KnowledgeFusion(d_model, fp_name=fp_name) for fp_name in knodes])
    
    def forward(self, bg, x, knodes):
        for i, fp_name in enumerate(self.knodes):
            k_feature = knodes[fp_name] 
            x = self.k_fusions[i](bg, x, k_feature)
        return x 
        