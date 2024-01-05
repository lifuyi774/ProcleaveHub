import numpy as np
import torch
import torch.nn as nn
from gvp import GVP, GVPConvLayer, LayerNorm, tuple_index
from torch.distributions import Categorical
from torch_scatter import scatter_mean
from torch.nn import functional as F

class GVP_encoder(nn.Module):
    '''
    GVP-GNN for Model Quality Assessment as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param edge_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.1):
        
        super(GVP_encoder, self).__init__()
        activations = (F.relu, None)

        self.seq_in = seq_in
        if self.seq_in:
            self.W_s = nn.Embedding(20, 20) # seq to embedding 
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None), vector_gate=True)
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=True)
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, activations=activations, vector_gate=True, drop_rate=drop_rate)
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0), activations=activations, vector_gate=True))
            
        self.dense = nn.Sequential(
            nn.Linear(ns, ns), nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            #nn.Linear(2*ns, 1)
        )

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        if self.seq_in and seq is not None:
            seq = self.W_s(seq)
            h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1]) 

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)

        out = self.W_out(h_V)
        
        if batch is None: out = out.mean(dim=0, keepdims=True)
        else: out = scatter_mean(out, batch, dim=0)
        
        return self.dense(out) 


class GLM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, augment_eps, dropout):
        super(GLM, self).__init__()

        self.GVP_encoder = GVP_encoder(node_in_dim=(input_dim,3), node_h_dim=(hidden_dim, 16), edge_in_dim=(32,1), edge_h_dim=(32, 1), seq_in=True, num_layers=num_layers, drop_rate=dropout)
        self.FC_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_2 = nn.Linear(hidden_dim, 2, bias=True)
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, data):
        h_V = (data.node_s, data.node_v)
        h_E = (data.edge_s, data.edge_v)
        edge_index,seq,batch=data.edge_index,data.seq,data.batch
        h_V = self.GVP_encoder(h_V, edge_index, h_E, seq,batch) 
        out=[]
        logits = self.FC_2(F.elu(self.FC_1(h_V))) 
        out.append(logits)
        return out

class PBMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout,batch_size):
        super(PBMLP, self).__init__()
        self.W_s = nn.Embedding(21, 20)
        self.dropout = nn.Dropout(dropout)
        self.kernel_size=5
        self.pool_size=5

        self.FC_1 = nn.Linear(input_dim*20, hidden_dim, bias=True)
        self.FC_2 = nn.Linear(hidden_dim, 2, bias=True)
        self.batch_size=batch_size
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x,seq):

        seq = self.W_s(seq)
        x=torch.cat([x, seq], dim=-1)
        x=x.view(x.shape[0], -1)

        x=self.dropout(x)
        out=[]

        logits = self.FC_2(F.elu(self.FC_1(x)))
        out.append(logits)
        return out

class GVPEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, augment_eps, dropout):#1044, 128, 5, 0.95, 0.1
        super(GVPEnsemble, self).__init__()

        self.GVP_encoder1 = GVP_encoder(node_in_dim=(input_dim,3), node_h_dim=(hidden_dim, 16), edge_in_dim=(32,1), edge_h_dim=(32, 1), seq_in=True, num_layers=num_layers, drop_rate=dropout)
        self.GVP_encoder2 = GVP_encoder(node_in_dim=(input_dim,3), node_h_dim=(hidden_dim, 16), edge_in_dim=(32,1), edge_h_dim=(32, 1), seq_in=True, num_layers=num_layers, drop_rate=dropout)
        self.GVP_encoder3 = GVP_encoder(node_in_dim=(input_dim,3), node_h_dim=(hidden_dim, 16), edge_in_dim=(32,1), edge_h_dim=(32, 1), seq_in=True, num_layers=num_layers, drop_rate=dropout)

        self.FC_1 = nn.Linear(hidden_dim*3, hidden_dim*3, bias=True)
        self.FC_2 = nn.Linear(hidden_dim*3, 2, bias=True)
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, data,data1,data2):
        
        h_V1 = (data.node_s, data.node_v)
        h_E1 = (data.edge_s, data.edge_v)
        edge_index1,seq1,batch1=data.edge_index,data.seq,data.batch
        h_V2 = (data1.node_s, data1.node_v)
        h_E2 = (data1.edge_s, data1.edge_v)
        edge_index2,seq2,batch2=data1.edge_index,data1.seq,data1.batch
        h_V3 = (data2.node_s, data2.node_v)
        h_E3 = (data2.edge_s, data2.edge_v)
        edge_index3,seq3,batch3=data2.edge_index,data2.seq,data2.batch
        
        h_V1 = self.GVP_encoder1(h_V1, edge_index1, h_E1, seq1,batch1) 
        h_V2= self.GVP_encoder2(h_V2, edge_index2, h_E2, seq2,batch2)
        h_V3 = self.GVP_encoder3(h_V3, edge_index3, h_E3, seq3,batch3)

        out=[]
        cat_v=torch.cat((h_V1, h_V2,h_V3), 1)
        logits = self.FC_2(F.elu(self.FC_1(cat_v))) 
        out.append(logits)
        return out
