import numpy as np
import random
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
import os
import features



def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, shape [...dims],if `D` has  then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _dihedrals(X, eps=1e-7):
    
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2]) 
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features
    
    
def _positional_embeddings(edge_index, num_embeddings=None, period_range=[2, 1000],device='cpu',num_positional_embeddings=16):
    num_embeddings = num_embeddings or num_positional_embeddings
    d = edge_index[0] - edge_index[1]
    
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E

def _orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def _sidechains(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec 
    
def _get_geo_edge_feat( X_ca, edge_index):
    u = torch.ones_like(X_ca)
    u[1:] = X_ca[1:] - X_ca[:-1]
    u = F.normalize(u, dim=-1)
    b = torch.ones_like(X_ca)
    b[:-1] = u[:-1] - u[1:]
    b = F.normalize(b, dim=-1)
    n = torch.ones_like(X_ca)
    n[:-1] = torch.cross(u[:-1], u[1:])
    n = F.normalize(n, dim=-1)

    local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1) # [L, 3, 3]

    node_j, node_i = edge_index
    t = F.normalize(X_ca[node_j] - X_ca[node_i], dim=-1)
    t = torch.einsum('ijk,ij->ik', local_frame[node_i], t) # [E, 3]
    #r = torch.sum(local_frame[node_i] * local_frame[node_j], dim=1)
    r = torch.matmul(local_frame[node_i].transpose(-1,-2), local_frame[node_j]) # [E, 3, 3]
    Q = _quaternions(r) # [E, 4]

    return torch.cat([t, Q], dim=-1) # [E, 3 + 4]
    
def _quaternions(R):
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)

    return Q

def SequentialEdge(x,device):
    # row=torch.tensor([v for v in range(x.shape[0]-1)])
    # col=torch.tensor([v+1 for v in range(x.shape[0]-1)])
    row=torch.as_tensor([v for v in range(x.shape[0]-1)], device=device, dtype=torch.long)
    col=torch.as_tensor([v+1 for v in range(x.shape[0]-1)], device=device, dtype=torch.long)
    # print(torch.stack([row, col], dim=0))#x[torch.stack([row, col], dim=0)[0]] - x[torch.stack([row, col], dim=0)[1]]
    # print(x)
    # print(torch.stack([row, col], dim=0)[0])
    # print(x[torch.stack([row, col], dim=0)[0]])
    return torch.stack([row, col], dim=0)
def radiusEdge(X_ca,r,mnn):
    return torch_cluster.radius_graph(X_ca, r=r,max_num_neighbors=mnn)
def knnEdge(X_ca,top_k):
    return torch_cluster.knn_graph(X_ca, k=top_k) # 利用Ca 原子坐标，获取k近邻来构图，

class BatchSampler(data.Sampler):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design.
    
    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.
    
    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(self, node_counts, max_nodes=3000, shuffle=True):
        
        self.node_counts = node_counts
        self.idx = [i for i in range(len(node_counts))  
                        if node_counts[i] <= max_nodes]
        self.shuffle = shuffle
        self.max_nodes = max_nodes
        self._form_batches()
    
    def _form_batches(self):
        self.batches = []
        if self.shuffle: random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)
    
    def __len__(self): 
        if not self.batches: self._form_batches()
        return len(self.batches)
    
    def __iter__(self):
        if not self.batches: self._form_batches()
        for batch in self.batches: yield batch


class ProteinGraphDataset(data.Dataset):
    '''
    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary-style
    protein structures into featurized protein graphs as described in the 
    manuscript.
    
    Returned graphs are of type `torch_geometric.data.Data` with attributes
    -x          alpha carbon coordinates, shape [n_nodes, 3]
    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
    -name       name of the protein structure, string
    -node_s     node scalar features, shape [n_nodes, 6] 
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, 32]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    -edge_index edge indices, shape [2, n_edges]
    -mask       node mask, `False` for nodes with missing data that are excluded from message passing
    
    Portions from https://github.com/jingraham/neurips19-graph-protein-design.
    
    :param data_list: JSON/dictionary-style protein dataset as described in README.md.
    :param num_positional_embeddings: number of positional embeddings
    :param top_k: number of edges to draw per node (as destination node)
    :param device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, dataset, index, fasta_file,args, 
                 num_positional_embeddings=16,graph_type='knn',
                 top_k=20, num_rbf=16, device="cpu"): #task_list,
        
        super(ProteinGraphDataset, self).__init__()

        self.dataset = {}
        index = set(index)
        for i, ID in enumerate(dataset):
            if i in index:
                self.dataset[ID] = dataset[ID]
        self.IDs = list(self.dataset.keys())
        

        self.posDict={}
        self.yDict={}
        with open(fasta_file) as r1:
            fasta_ori = r1.readlines()
        for i in range(len(fasta_ori)):
            # if fasta_ori[i][0] == ">":
            #     name = fasta_ori[i].split('>')[1].replace('\n','') 
                
            #     seq = fasta_ori[i+1].replace('\n','')
            #     pdbfasta[name] = seq
            if fasta_ori[i][0] == ">":
                # name = fasta_ori[i].split('>')[1].replace('\n','') 
                descrL=fasta_ori[i].split('>')[1].replace('\n','').split()
                y=int(descrL[1])
                pos=int(descrL[0].split('&')[1])
                
                # Uid=descrL[0].split('&')[0].split('_')[-1]# uniprotid
                name=descrL[0]
                self.posDict[name]=pos
                self.yDict[name]=y
        ######


        self.dataset_path = args.dataset_path
        self.feature_path = args.feature_path
        self.fasta_file = fasta_file
        self.pdb_path =args.pdb_path 
        self.output_prottrans = args.output_prottrans
        self.output_esmfold = args.output_esmfold
        self.output_dssp = args.output_dssp
        # self.residue_feature_path=args.residue_feature_path
        self.output_residue = args.output_dssp#args.output_residue
        # self.task_list = task_list
        self.graph_type=graph_type
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(self.dataset[ID][2]) for ID in self.IDs]
        
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        self.protease=args.protease
        self.chain=args.chain
        
    def __len__(self): return len(self.IDs)
    
    def __getitem__(self, idx): return self._featurize_as_graph(idx)
    
    def _featurize_as_graph(self, idx):
        name = self.IDs[idx]
        # Uid=name.split('&')[0].split('_')[-1]# uniprotid
        
        
        pos=self.posDict[name]
        if (pos-10)<0:
            s=0
        else:
            s=pos-10
        e=pos+10
        #########
        with torch.no_grad():
            if not os.path.exists(self.feature_path + name + ".npy"):
                # print("run_ptottrans")
                features.get_prottrans(self.fasta_file, self.output_prottrans) 
            if not os.path.exists(self.dataset_path + name + ".npy"):
                # print('run_esmfold1')
                # features.get_esmfold(self.fasta_file, self.output_esmfold)
                features.get_coord_feature_for_train(self.fasta_file, self.output_esmfold, self.pdb_path,self.protease,self.chain)
            if not os.path.exists(self.dataset_path + name + "_dssp.npy"):
                # print('run_get_dssp')
                # features.get_dssp(self.fasta_file, self.output_esmfold, self.output_dssp)
                features.get_dssp_for_train(self.fasta_file, self.pdb_path, self.output_dssp,self.protease,self.chain)
            # if not os.path.exists(self.dataset_path + name + "_residue.npy"):
            #     # features.get_dssp(self.fasta_file, self.output_esmfold, self.output_dssp)
            #     features.get_residue_feature(self.fasta_file, 'C14.001',self.residue_feature_path, self.output_residue)
            
            # try:
            #     coords = np.load(self.dataset_path + name + ".npy")#,allow_pickle=True
            # except:
            #     print(name)
            coords = np.load(self.dataset_path + name + ".npy")#,allow_pickle=True
            # AAtype = np.load(self.dataset_path + name + "_aatype_.npy")
            coords = torch.as_tensor(coords, device=self.device, dtype=torch.float32)
            
            
            # prottrans_feat = torch.load(self.feature_path + name + ".tensor")
            # prottrans_feat = torch.tensor(np.load(self.feature_path + name + ".npy")) 
            # try:
            #     prottrans_feat = torch.tensor(np.load(self.feature_path + name + ".npy"))
            # except:
            #     print(name)
            prottrans_feat = torch.tensor(np.load(self.feature_path + name + ".npy")) 
            # dssp = torch.tensor(np.load(self.dataset_path + name + "_dssp.npy"))
            dssp = torch.tensor(np.load(self.dataset_path + name + "_dssp.npy"))
            # print(name)
            # residueF = torch.tensor(np.load(self.dataset_path + name + "_residue.npy",allow_pickle=True).astype(float))
            # print(name,dihedrals.shape,prottrans_feat.shape,dssp.shape)
            # print(dihedrals.shape,prottrans_feat.shape,residueF.shape)
            
            # mask = torch.isfinite(coords.sum(dim=(1,2)))
            # coords[~mask] = np.inf
            X_ca = coords[:, 1] 
            # print("X_ca shape",X_ca.shape,name,pos)
            if self.graph_type =='knn' or self.graph_type =='rball':
                index_coord_distance=[]
                for i in range(len(X_ca)):

                    d=torch.dist(X_ca[pos-1], X_ca[i])
                    if d<=10: 
                        index_coord_distance.append(i)
               
                X_ca=X_ca[index_coord_distance]
                coords=coords[index_coord_distance]
                prottrans_feat=prottrans_feat[index_coord_distance]
                dssp=dssp[index_coord_distance]
                seq_new=''
                for i in index_coord_distance:
                    seq_new+=self.dataset[name][2][i]
            else:
                X_ca=X_ca[s:e]
                coords=coords[s:e]
                prottrans_feat=prottrans_feat[s:e]
                dssp=dssp[s:e]
                seq_new=self.dataset[name][2][s:e]
            # AA_type_ca=AAtype[:,1]
            seq = torch.as_tensor([self.letter_to_num[aa] for aa in seq_new],
                                  device=self.device, dtype=torch.long) 
            if self.graph_type =='knn':
                edge_index = knnEdge(X_ca, self.top_k)
            elif self.graph_type =='rball':
                edge_index = radiusEdge(X_ca,8,10)
            else:
                edge_index = SequentialEdge(X_ca,device=self.device)
            
            pos_embeddings = _positional_embeddings(edge_index,device=self.device,num_positional_embeddings=self.num_positional_embeddings)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            # AA_type1=AA_type_ca[edge_index[0]]
            # AA_type2=AA_type_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            geo_edge_feat = _get_geo_edge_feat(X_ca, edge_index)
            dihedrals = _dihedrals(coords)                     
            orientations = _orientations(X_ca)
            sidechains = _sidechains(coords)
            
            
            # print(name,dssp.shape,dihedrals.shape,prottrans_feat)
            node_s = torch.cat([dihedrals, prottrans_feat, dssp], dim=-1).to(torch.float32)
            # node_s = torch.cat([dihedrals, prottrans_feat, residueF], dim=-1).to(torch.float32)

            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)
            
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                    (node_s, node_v, edge_s, edge_v))
            
            y = [self.yDict[name]]

            # y_mask = []
            # for i in range(len(self.task_list)):
            #     # y_task = self.dataset[name][1][i]
                
            #     if y_task:
            #         y.append(y_task)
            #         y_mask.append([1] * len(seq))
            #        
            #     else:
            #         y.append([0] * len(seq))
            #         y_mask.append([0] * len(seq))
    
            y = torch.as_tensor(y, device=self.device, dtype=torch.float32)#.t()
            # y_mask = torch.as_tensor(y_mask, device=self.device, dtype=torch.float32).t()

        data = torch_geometric.data.Data(x=X_ca, seq=seq, name=name,
                                         node_s=node_s, node_v=node_v,
                                         edge_s=edge_s, edge_v=edge_v,
                                         edge_index=edge_index,
                                         y = y)#, y_mask = y_mask, mask=mask,
        return data
                                
# For prediction
class ProteinGraphDataset2(data.Dataset):
    '''
    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary-style
    protein structures into featurized protein graphs as described in the 
    manuscript.
    
    Returned graphs are of type `torch_geometric.data.Data` with attributes
    -x          alpha carbon coordinates, shape [n_nodes, 3]
    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
    -name       name of the protein structure, string
    -node_s     node scalar features, shape [n_nodes, 6] 
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, 32]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    -edge_index edge indices, shape [2, n_edges]
    -mask       node mask, `False` for nodes with missing data that are excluded from message passing
    
    Portions from https://github.com/jingraham/neurips19-graph-protein-design.
    
    :param data_list: JSON/dictionary-style protein dataset as described in README.md.
    :param num_positional_embeddings: number of positional embeddings
    :param top_k: number of edges to draw per node (as destination node)
    :param device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, dataset, index, fasta_file,args, 
                 num_positional_embeddings=16,graph_type='knn',
                 top_k=20, num_rbf=16, device="cpu"): #task_list,
        
        super(ProteinGraphDataset2, self).__init__()

        self.dataset = {}
        index = set(index)
        for i, ID in enumerate(dataset):
            if i in index:
                self.dataset[ID] = dataset[ID]
        self.IDs = list(self.dataset.keys())
        
        # 加，提取子序列的特征
        self.posDict={}
        self.yDict={}
        for key,value in dataset.items():
            self.posDict[key]=value[0]
            self.yDict[key]=value[1]


        self.chain=args.chain
        self.dataset_path = args.dataset_path
        self.feature_path = args.feature_path
        self.fasta_file = fasta_file

        self.pdb_path =args.pre_file # 加 预测模块时输入PDB ，inputpath 是pdb路径
        self.output_prottrans = args.output_prottrans
        self.output_esmfold = args.output_esmfold
        self.output_dssp = args.output_dssp
        # self.residue_feature_path=args.residue_feature_path
        self.output_residue = args.output_dssp#args.output_residue
        # self.task_list = task_list
        self.graph_type=graph_type
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(self.dataset[ID][2]) for ID in self.IDs]
        
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12,'X':20}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        self.protease=args.protease
        
        
    def __len__(self): return len(self.IDs)
    
    def __getitem__(self, idx): return self._featurize_as_graph(idx)
    
    def _featurize_as_graph(self, idx):
        name = self.IDs[idx]
        # Uid=name.split('&')[0].split('_')[-1]# uniprotid
        pdbid=name.split('&')[0]
        # 提取子序列的dssp feature 加
        pos=self.posDict[name]
        if (pos-10)<0:
            s=0
        else:
            s=pos-10
        e=pos+10
        #########
        with torch.no_grad():
            if not os.path.exists(self.feature_path + pdbid + ".npy"): #计算子序列的嵌入表示
                # print("run_ptottrans")
                features.get_prottrans1(self.fasta_file, self.output_prottrans) #输入改为长度为20的序列
            if not os.path.exists(self.dataset_path + pdbid + ".npy"): # 计算的coor 是全长序列的，下面需提取子序列的coord
                # print('run_esmfold1')
                # features.get_esmfold(self.fasta_file, self.output_esmfold)
                # features.get_esmfold2(self.fasta_file, self.output_esmfold, self.pdb_path,self.protease,self.chain)
                features.get_coord_feature_for_pre(self.fasta_file, self.output_esmfold, self.pdb_path,self.protease,self.chain)
            if not os.path.exists(self.dataset_path + pdbid + "_dssp.npy"):# 计算的是全长序列对应的dssp 特征，下面需要提取子序列的dssp 特征
                # print('run_get_dssp')
                # features.get_dssp(self.fasta_file, self.output_esmfold, self.output_dssp)
                features.get_dssp_for_pre(self.fasta_file, self.pdb_path, self.output_dssp,self.protease,self.chain)
            # if not os.path.exists(self.dataset_path + name + "_residue.npy"):
            #     # features.get_dssp(self.fasta_file, self.output_esmfold, self.output_dssp)
            #     features.get_residue_feature(self.fasta_file, 'C14.001',self.residue_feature_path, self.output_residue)
            
            # try:
            #     coords = np.load(self.dataset_path + name + ".npy")#,allow_pickle=True
            # except:
            #     print(name)
            coords = np.load(self.dataset_path + pdbid + ".npy")#,allow_pickle=True
            # AAtype = np.load(self.dataset_path + name + "_aatype_.npy")
            coords = torch.as_tensor(coords, device=self.device, dtype=torch.float32)
            
            
            # prottrans_feat = torch.load(self.feature_path + name + ".tensor")
            # prottrans_feat = torch.tensor(np.load(self.feature_path + name + ".npy")) # 如何获取子序列的 prottrans? 需要改, (n,1024)
            # try:
            #     prottrans_feat = torch.tensor(np.load(self.feature_path + name + ".npy"))
            # except:
            #     print(name)
            prottrans_feat = torch.tensor(np.load(self.feature_path + pdbid + ".npy")) # 如何获取子序列的 prottrans? 需要改, (n,1024)
            # dssp = torch.tensor(np.load(self.dataset_path + name + "_dssp.npy"))
            dssp = torch.tensor(np.load(self.dataset_path + pdbid + "_dssp.npy"))
            # print(name)
            # residueF = torch.tensor(np.load(self.dataset_path + name + "_residue.npy",allow_pickle=True).astype(float))
            # print(name,dihedrals.shape,prottrans_feat.shape,dssp.shape)
            # print(dihedrals.shape,prottrans_feat.shape,residueF.shape)
            
            # mask = torch.isfinite(coords.sum(dim=(1,2)))
            # coords[~mask] = np.inf
            X_ca = coords[:, 1] #取的Ca的坐标
            # print("X_ca shape",X_ca.shape,name,pos)
            if self.graph_type =='knn' or self.graph_type =='rball':
                index_coord_distance=[]
                for i in range(len(X_ca)):
                    # 1. 遍历所有ca coord,找到与 P1-CA 原子距离小于10A的所有原子对应的索引。
                    # d=distance.euclidean(tuple(list(X_ca[p-1])), tuple(list(X_ca[i])))
                    # try:
                    #     d=torch.dist(X_ca[pos-1], X_ca[i])
                    # except:
                    #     print("X_ca shape",X_ca.shape,name,pos)
                    d=torch.dist(X_ca[pos-1], X_ca[i])
                    if d<=10: 
                        index_coord_distance.append(i)
                # 2. 利用索引从全部的coords数组中提取子coords数组
                X_ca=X_ca[index_coord_distance]
                coords=coords[index_coord_distance]
                prottrans_feat=prottrans_feat[index_coord_distance]
                dssp=dssp[index_coord_distance]
                seq_new=''
                for i in index_coord_distance:
                    seq_new+=self.dataset[name][2][i] # dataset[name] (pos,y,seq)
            else:
                X_ca=X_ca[s:e]
                coords=coords[s:e]
                prottrans_feat=prottrans_feat[s:e]
                dssp=dssp[s:e]
                seq_new=self.dataset[name][2][s:e]
            # AA_type_ca=AAtype[:,1]
            seq = torch.as_tensor([self.letter_to_num[aa] for aa in seq_new],
                                  device=self.device, dtype=torch.long) # self.dataset[name][0]  输入序列长度改为20
            # 构图 加边
            # edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k) # 利用Ca 原子坐标，获取k近邻来构图，每个节点的30个邻接
            # edge_index = torch_cluster.radius_graph(X_ca, r=8,max_num_neighbors=10)
            if self.graph_type =='knn':
                edge_index = knnEdge(X_ca, self.top_k)
            elif self.graph_type =='rball':
                edge_index = radiusEdge(X_ca,8,10)
            else:
                edge_index = SequentialEdge(X_ca,device=self.device) # Sequential graph
            
            pos_embeddings = _positional_embeddings(edge_index,device=self.device,num_positional_embeddings=self.num_positional_embeddings)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            # AA_type1=AA_type_ca[edge_index[0]]
            # AA_type2=AA_type_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            geo_edge_feat = _get_geo_edge_feat(X_ca, edge_index)
            dihedrals = _dihedrals(coords)                     
            orientations = _orientations(X_ca)
            sidechains = _sidechains(coords)
            
            
            # print(name,dssp.shape,dihedrals.shape,prottrans_feat)
            node_s = torch.cat([dihedrals, prottrans_feat, dssp], dim=-1).to(torch.float32)
            # node_s = torch.cat([dihedrals, prottrans_feat, residueF], dim=-1).to(torch.float32)

            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)
            
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                    (node_s, node_v, edge_s, edge_v))
            
            y = [self.yDict[name]]

            # y_mask = []
            # for i in range(len(self.task_list)):
            #     # y_task = self.dataset[name][1][i]
                
            #     if y_task:
            #         y.append(y_task)
            #         y_mask.append([1] * len(seq))
            #        
            #     else:
            #         y.append([0] * len(seq))
            #         y_mask.append([0] * len(seq))
    
            y = torch.as_tensor(y, device=self.device, dtype=torch.float32)#.t()
            # y_mask = torch.as_tensor(y_mask, device=self.device, dtype=torch.float32).t()

        data = torch_geometric.data.Data(x=X_ca, seq=seq, name=name,
                                         node_s=node_s, node_v=node_v,
                                         edge_s=edge_s, edge_v=edge_v,
                                         edge_index=edge_index,
                                         y = y)#, y_mask = y_mask, mask=mask,
        return data


class ProteinSequenceDataset(data.Dataset):
    '''
    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary-style
    protein structures into featurized protein graphs as described in the 
    manuscript.
    
    Returned graphs are of type `torch_geometric.data.Data` with attributes
    -x          alpha carbon coordinates, shape [n_nodes, 3]
    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
    -name       name of the protein structure, string
    -node_s     node scalar features, shape [n_nodes, 6] 
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, 32]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    -edge_index edge indices, shape [2, n_edges]
    -mask       node mask, `False` for nodes with missing data that are excluded from message passing
    
    Portions from https://github.com/jingraham/neurips19-graph-protein-design.
    
    :param data_list: JSON/dictionary-style protein dataset as described in README.md.
    :param num_positional_embeddings: number of positional embeddings
    :param top_k: number of edges to draw per node (as destination node)
    :param device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, dataset, index, fasta_file,args, 
                 num_positional_embeddings=16,graph_type='knn',
                 top_k=20, num_rbf=16, device="cpu"): #task_list,
        
        super(ProteinSequenceDataset, self).__init__()

        self.dataset = {}
        index = set(index)
        for i, ID in enumerate(dataset):
            if i in index:
                self.dataset[ID] = dataset[ID]
        self.IDs = list(self.dataset.keys())
        
        # 加，提取子序列的特征
        self.posDict={}
        self.yDict={}
        with open(fasta_file) as r1:
            fasta_ori = r1.readlines()
        for i in range(len(fasta_ori)):
            # if fasta_ori[i][0] == ">":
            #     name = fasta_ori[i].split('>')[1].replace('\n','') # 增加 分割name的代码
                
            #     seq = fasta_ori[i+1].replace('\n','')
            #     pdbfasta[name] = seq
            if fasta_ori[i][0] == ">":
                # name = fasta_ori[i].split('>')[1].replace('\n','') # 增加 分割name的代码
                descrL=fasta_ori[i].split('>')[1].replace('\n','').split()
                y=int(descrL[1])
                pos=int(descrL[0].split('&')[1])
                
                # Uid=descrL[0].split('&')[0].split('_')[-1]# uniprotid
                name=descrL[0]
                self.posDict[name]=pos
                self.yDict[name]=y
        ######


        self.dataset_path = args.dataset_path
        self.feature_path = args.feature_path
        self.fasta_file = fasta_file
        self.pdb_path =args.pdb_path # 加
        self.output_prottrans = args.output_prottrans
        self.output_esmfold = args.output_esmfold
        self.output_dssp = args.output_dssp
        # self.residue_feature_path=args.residue_feature_path
        self.output_residue = args.output_dssp#args.output_residue
        # self.task_list = task_list
        self.graph_type=graph_type
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(self.dataset[ID][2]) for ID in self.IDs]
        
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12,'X':20}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        self.protease=args.protease
        
        
    def __len__(self): return len(self.IDs)
    
    def __getitem__(self, idx): 
        name = self.IDs[idx]
        y = [self.yDict[name]]
        y = torch.as_tensor(y, device=self.device, dtype=torch.long)
        # print(y.shape)
        prottrans_feat,seq=self._featurize_as_embed(idx)
        return prottrans_feat,seq,y
    
    def _featurize_as_embed(self, idx):
        name = self.IDs[idx]
        # Uid=name.split('&')[0].split('_')[-1]# uniprotid
        
        # 提取子序列的dssp feature 加
        pos=self.posDict[name]
        if (pos-10)<0:
            s=0
        else:
            s=pos-10
        e=pos+10
        #########
        seq_all=self.dataset[name][2]
        seq_cut=seq_all[s:e]
        with torch.no_grad():
            if not os.path.exists(self.feature_path + name + ".npy"):
                # print("run_ptottrans")
                features.get_prottrans(self.fasta_file, self.output_prottrans) #输入改为长度为20的序列
            prottrans_feat = torch.tensor(np.load(self.feature_path + name + ".npy")) # 如何获取子序列的 prottrans? 需要改, (n,1024)
            prottrans_feat=prottrans_feat[s:e]#.view(-1)
            if (pos-10)<0:# 表示上游不足10，需要补充
                l=10-pos #需要补充
                seq_cut='X'*l+seq_cut
                zeroTensor=torch.zeros([l,1024],dtype=torch.float32)
                prottrans_feat=torch.cat([zeroTensor, prottrans_feat], dim=0) # 对'X'对应的特征用全0 pading
            elif (len(seq_all)-pos)<10: # 下游不足10 pading
                l=10-(len(seq_all)-pos)
                seq_cut=seq_cut+'X'*l
                zeroTensor=torch.zeros([l,1024],dtype=torch.float32)
                prottrans_feat=torch.cat([prottrans_feat,zeroTensor], dim=0)

            seq = torch.as_tensor([self.letter_to_num[aa] for aa in seq_cut],
                                  device=self.device, dtype=torch.long)
        return prottrans_feat,seq

class ProteinSequenceDataset1(data.Dataset):
    '''
    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary-style
    protein structures into featurized protein graphs as described in the 
    manuscript.
    
    Returned graphs are of type `torch_geometric.data.Data` with attributes
    -x          alpha carbon coordinates, shape [n_nodes, 3]
    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
    -name       name of the protein structure, string
    -node_s     node scalar features, shape [n_nodes, 6] 
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, 32]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    -edge_index edge indices, shape [2, n_edges]
    -mask       node mask, `False` for nodes with missing data that are excluded from message passing
    
    Portions from https://github.com/jingraham/neurips19-graph-protein-design.
    
    :param data_list: JSON/dictionary-style protein dataset as described in README.md.
    :param num_positional_embeddings: number of positional embeddings
    :param top_k: number of edges to draw per node (as destination node)
    :param device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, dataset, index, fasta_file,args, 
                 num_positional_embeddings=16,graph_type='knn',
                 top_k=20, num_rbf=16, device="cpu"): #task_list,
        
        super(ProteinSequenceDataset1, self).__init__()

        self.dataset = {}
        index = set(index)
        for i, ID in enumerate(dataset):
            if i in index:
                self.dataset[ID] = dataset[ID]
        self.IDs = list(self.dataset.keys())
        
        # 加，提取子序列的特征
        self.posDict={}
        self.yDict={}
        for key,value in dataset.items():
            self.posDict[key]=value[0]
            self.yDict[key]=value[1]
        # with open(fasta_file) as r1:
        #     fasta_ori = r1.readlines()
        # for i in range(len(fasta_ori)):
        #     # if fasta_ori[i][0] == ">":
        #     #     name = fasta_ori[i].split('>')[1].replace('\n','') # 增加 分割name的代码
                
        #     #     seq = fasta_ori[i+1].replace('\n','')
        #     #     pdbfasta[name] = seq
        #     if fasta_ori[i][0] == ">":
        #         # name = fasta_ori[i].split('>')[1].replace('\n','') # 增加 分割name的代码
        #         descrL=fasta_ori[i].split('>')[1].replace('\n','').split()
        #         y=int(descrL[1])
        #         pos=int(descrL[0].split('&')[1])
                
        #         # Uid=descrL[0].split('&')[0].split('_')[-1]# uniprotid
        #         name=descrL[0]
        #         self.posDict[name]=pos
        #         self.yDict[name]=y
        ######


        self.dataset_path = args.dataset_path
        self.feature_path = args.feature_path
        self.fasta_file = fasta_file
        # self.pdb_path =args.pdb_path # 加
        self.output_prottrans = args.output_prottrans
        self.output_esmfold = args.output_esmfold
        self.output_dssp = args.output_dssp
        # self.residue_feature_path=args.residue_feature_path
        self.output_residue = args.output_dssp#args.output_residue
        # self.task_list = task_list
        self.graph_type=graph_type
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(self.dataset[ID][2]) for ID in self.IDs]
        
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12,'X':20}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        self.protease=args.protease
        
        
    def __len__(self): return len(self.IDs)
    
    def __getitem__(self, idx): 
        name = self.IDs[idx]
        y = [self.yDict[name]]
        y = torch.as_tensor(y, device=self.device, dtype=torch.long)
        # print(y.shape)
        prottrans_feat,seq=self._featurize_as_embed(idx)
        return prottrans_feat,seq,y
    
    def _featurize_as_embed(self, idx):
        
        name = self.IDs[idx]
        pdbid=name.split('&')[0]
        # Uid=name.split('&')[0].split('_')[-1]# uniprotid
        
        # 提取子序列的dssp feature 加
        pos=self.posDict[name]
        if (pos-10)<0:
            s=0
        else:
            s=pos-10
        e=pos+10
        #########
        seq_all=self.dataset[name][2]
        seq_cut=seq_all[s:e]
        with torch.no_grad():
            if not os.path.exists(self.feature_path + pdbid + ".npy"):
                # print("run_ptottrans")
                features.get_prottrans1(self.fasta_file, self.output_prottrans) #输入改为长度为20的序列
            prottrans_feat = torch.tensor(np.load(self.feature_path + pdbid + ".npy")) # 如何获取子序列的 prottrans? 需要改, (n,1024)
            prottrans_feat=prottrans_feat[s:e]#.view(-1)
            if (pos-10)<0:# 表示上游不足10，需要补充
                l=10-pos #需要补充
                seq_cut='X'*l+seq_cut
                zeroTensor=torch.zeros([l,1024],dtype=torch.float32)
                prottrans_feat=torch.cat([zeroTensor, prottrans_feat], dim=0) # 对'X'对应的特征用全0 pading
            elif (len(seq_all)-pos)<10: # 下游不足10 pading
                l=10-(len(seq_all)-pos)
                seq_cut=seq_cut+'X'*l
                zeroTensor=torch.zeros([l,1024],dtype=torch.float32)
                prottrans_feat=torch.cat([prottrans_feat,zeroTensor], dim=0)

            seq = torch.as_tensor([self.letter_to_num[aa] for aa in seq_cut],
                                  device=self.device, dtype=torch.long)
        return prottrans_feat,seq
