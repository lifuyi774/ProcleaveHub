import torch
from transformers import T5EncoderModel, T5Tokenizer
import re, argparse
import numpy as np
from tqdm import tqdm
import gc
import multiprocessing
import os, datetime
from Bio import pairwise2
import pickle
import math
import pandas as pd
from typing import Any, Dict, Optional
from Bio.Data.IUPACData import protein_letters_1to3
from Bio import PDB
# from Bio.PDB.SASA import ShrakeRupley
# from Bio.PDB import DSSP
# from Bio.PDB import HSExposureCB
from Bio.PDB import NACCESS
AA_indx = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
                   'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, '-': 0,'X': 0}
one_letter={'VAL': 'V','ILE': 'I','LEU': 'L','GLU': 'E','GLN': 'Q','ASP': 'D','ASN': 'N','HIS': 'H','TRP': 'W',\
    'PHE': 'F','TYR': 'Y','ARG': 'R','LYS': 'K','SER': 'S','THR': 'T','MET': 'M','ALA': 'A','GLY': 'G','PRO': 'P','CYS': 'C'}

def get_prottrans(fasta_file,output_path):
    
    num_cores = 2
    multiprocessing.set_start_method("forkserver", force=True)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    
    # parser1 = argparse.ArgumentParser()
    # parser1.add_argument("--gpu", type = str, default = '0')
    # args1 = parser1.parse_args()
    # gpu = args1.gpu
    gpu ='0'
   
    ID_list = []
    seq_list = []
    with open(fasta_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == ">":
            # ID_list.append(line[1:-1])
            descrL=line.split('>')[1].replace('\n','').split()
            # Uid=descrL[0].split('&')[0].split('_')[-1]# uniprotid
            # ID_=descrL[0].split('&')[0]
            ID_=descrL[0]
            ID_list.append(ID_)
        else:

            seq_list.append(" ".join(list(line.strip())))

    model_path = "../Prot_T5_XL_U50"
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path)
    gc.collect()
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')
    # device='cpu'
    model = model.eval()
    model = model.cuda()
    # print(next(model.parameters()).device)
    # print('starttime')
    # starttime = datetime.datetime.now()
    # print(starttime)
    batch_size = 1

    for i in tqdm(range(0, len(ID_list), batch_size)):
        if i + batch_size <= len(ID_list):
            batch_ID_list = ID_list[i:i + batch_size]
            batch_seq_list = seq_list[i:i + batch_size]
        else:
            batch_ID_list = ID_list[i:]
            batch_seq_list = seq_list[i:]
        

        # Create or load sequences and map rarely occured amino acids (U,Z,O,B) to (X)
        batch_seq_list = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch_seq_list]

        # Tokenize, encode sequences and load it into the GPU if possibile
        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # Extracting sequences' features and load it into the CPU if needed
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        

        # Remove padding (\<pad>) and special tokens (\</s>) that is added by ProtT5-XL-UniRef50 model
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            np.save(output_path + "/" + batch_ID_list[seq_num], seq_emd)
            # endtime = datetime.datetime.now()
            # print('endtime')
            # print(endtime)  
            
def get_prottrans1(fasta_file,output_path):
    
    num_cores = 2
    multiprocessing.set_start_method("forkserver", force=True)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    
    # parser1 = argparse.ArgumentParser()
    # parser1.add_argument("--gpu", type = str, default = '0')
    # args1 = parser1.parse_args()
    # gpu = args1.gpu
    gpu ='0'
   
    ID_list = []
    seq_list = []
    with open(fasta_file, "r") as f:
        lines = f.readlines()
    for line in lines[0:2]:
        if line[0] == ">":
            # ID_list.append(line[1:-1])
            descrL=line.split('>')[1].replace('\n','').split()
            # Uid=descrL[0].split('&')[0].split('_')[-1]# uniprotid
            ID_=descrL[0].split('&')[0] #pdbid
            # ID_=descrL[0]
            ID_list.append(ID_)
        else:

            seq_list.append(" ".join(list(line.strip())))

    model_path = "../Prot_T5_XL_U50"
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path)
    gc.collect()
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')
    # device='cpu'
    model = model.eval()
    model = model.cuda()
    # print(next(model.parameters()).device)
    # print('starttime')
    # starttime = datetime.datetime.now()
    # print(starttime)
    batch_size = 1

    for i in tqdm(range(0, len(ID_list), batch_size)):
        if i + batch_size <= len(ID_list):
            batch_ID_list = ID_list[i:i + batch_size]
            batch_seq_list = seq_list[i:i + batch_size]
        else:
            batch_ID_list = ID_list[i:]
            batch_seq_list = seq_list[i:]
        

        # Create or load sequences and map rarely occured amino acids (U,Z,O,B) to (X)
        batch_seq_list = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch_seq_list]

        # Tokenize, encode sequences and load it into the GPU if possibile
        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # Extracting sequences' features and load it into the CPU if needed
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        

        # Remove padding (\<pad>) and special tokens (\</s>) that is added by ProtT5-XL-UniRef50 model
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            np.save(output_path + "/" + batch_ID_list[seq_num], seq_emd)
            # endtime = datetime.datetime.now()
            # print('endtime')
            # print(endtime)  

def calculate_num_chunks(sequence, max_chunk_size):
    total_length = len(sequence)
    num_chunks = (total_length + max_chunk_size - 1) // max_chunk_size
    return num_chunks

def calculate_chunk_size(sequence, num_chunks):
    total_length = len(sequence)
    avg_chunk_size = total_length // num_chunks
    remainder = total_length % num_chunks

    
    chunk_sizes = [avg_chunk_size] * num_chunks
    for i in range(remainder):
        chunk_sizes[i] += 1

    return chunk_sizes 
 

def get_pdb_xyz2(pdb_file,ref_seq,chain):
    with open(pdb_file) as r2:
        pdb_file_lines = r2.readlines()
    current_pos = -1000
    X = []
    # X1 =[]
    current_aa = {} # 'N', 'CA', 'C', 'O'
    # cuurent_aa1={}
    try:
        for line in pdb_file_lines:
            if 'ENDMDL' in line:
                break
            # if (line[0:4].strip() == "ATOM" and line[21].strip() =='B') or (line[0:4].strip() == "ATOM" and line[21].strip() =='C'): 
            #     break
            if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos and line[21].strip() ==chain) or line[0:4].strip() == "TER":
                if current_aa != {}:
                    X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"]])
                    # X1.append([current_aa1["N"], current_aa1["CA"], current_aa1["C"], current_aa1["O"]])
                    current_aa = {}
                    # current_aa1 = {}
                if line[0:4].strip() != "TER":
                    current_pos = int(line[22:26].strip())

            if line[0:4].strip() == "ATOM" and line[21].strip() ==chain:
                atom = line[13:16].strip()
                if atom in ['N', 'CA', 'C', 'O']:
                    xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                    current_aa[atom] = xyz
                    # aatype=np.arrat([AA_indx[one_letter[str(line[17:20].strip())]]]).astype(np.float32)
                    # cuurent_aa1[atom] = aatype
    except:
        return None
    
    # print('LEN X',len(X),len(ref_seq))
    if len(X) == len(ref_seq):          
        return np.array(X)#,np.array(X1)
    else:
        return None    
def get_pdb_xyz3(pdb_file,ref_seq,chain):
    with open(pdb_file) as r2:
        pdb_file_lines = r2.readlines()
    current_pos = -1000
    X = []
    # X1 =[]
    current_aa = {} # 'N', 'CA', 'C', 'O'
    # cuurent_aa1={}
    try:
        for line in pdb_file_lines:
            if 'ENDMDL' in line:
                break
            # if (line[0:4].strip() == "ATOM" and line[21].strip() =='B') or (line[0:4].strip() == "ATOM" and line[21].strip() =='C'): # 排除其它链
            #     break
            if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos and line[21].strip() ==chain) or line[0:4].strip() == "TER":
                if current_aa != {}:
                    X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"]])
                    # X1.append([current_aa1["N"], current_aa1["CA"], current_aa1["C"], current_aa1["O"]])
                    current_aa = {}
                    # current_aa1 = {}
                if line[0:4].strip() != "TER":
                    current_pos = int(line[22:26].strip())

            if line[0:4].strip() == "ATOM" and line[21].strip() ==chain:
                atom = line[13:16].strip()
                if atom in ['N', 'CA', 'C', 'O']:
                    xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                    current_aa[atom] = xyz
                    # aatype=np.arrat([AA_indx[one_letter[str(line[17:20].strip())]]]).astype(np.float32)
                    # cuurent_aa1[atom] = aatype
    except:
        return None
    # # # 提取子序列的dssp feature 加
    # if (pos-10)<0: # 10/15
    #     s=0
    # else:
    #     s=pos-10
    # e=pos+10
    # # #########
    # X=X[s:e]
    # print('LEN X',len(X),len(ref_seq))
    if len(X) == len(ref_seq):          
        return np.array(X)#,np.array(X1)
    else:
        return None 

def get_coord_feature_for_train(fasta_file,output_path,pdb_path,protease,chain):

    pdbfasta = {}
    # posDict={}
    with open(fasta_file) as r1:
        fasta_ori = r1.readlines()
    for i in range(len(fasta_ori)):
        if fasta_ori[i][0] == ">":
            descrL=fasta_ori[i].split('>')[1].replace('\n','').split()
            # y=int(descrL[1])
            # pos=int(descrL[0].split('&')[1])
            # Uid=descrL[0].split('&')[0].split('_')[-1]
            # if Uid in []:
            #     continue
            # else:
            name=descrL[0]
            seq = fasta_ori[i+1].replace('\n','')
            pdbfasta[name] = seq.replace('X','')
            # posDict[name]=pos

    for key in pdbfasta.keys():
        Uid=key.split('&')[0].split('_')[-1]
        coord = get_pdb_xyz2(pdb_path + '/' + Uid + '.pdb',pdbfasta[key],chain) 
        np.save(output_path  + key + '.npy', coord)

def get_coord_feature_for_pre(fasta_file,output_path,pdb_path,protease,chain):

    pdbfasta = {}
    # posDict={}
    with open(fasta_file) as r1:
        fasta_ori = r1.readlines()
    for i in range(len(fasta_ori)):
        if fasta_ori[i][0] == ">":
            # name = fasta_ori[i].split('>')[1].replace('\n','') # 增加 分割name的代码
            descrL=fasta_ori[i].split('>')[1].replace('\n','').split()
            # y=int(descrL[1])
            # pos=int(descrL[0].split('&')[1])
            # Uid=descrL[0].split('&')[0].split('_')[-1]# uniprotid
            # if Uid in []:
            #     continue
            # else:
            name=descrL[0]
            seq = fasta_ori[i+1].replace('\n','')
            pdbfasta[name] = seq.replace('X','')
            # posDict[name]=pos
    # print(pdbfasta)
    name=list(pdbfasta.keys())[0]
    coord = get_pdb_xyz3(pdb_path,pdbfasta[name],chain)
    # name=list(pdbfasta.keys())[0]
    pdbid=name.split('&')[0]
    
    np.save(output_path  + pdbid + '.npy', coord)



def get_dssp_for_train(fasta_file, pdb_path, dssp_path,protease,chain):
    DSSP = './dssp'
    def process_dssp2(dssp_file,chain):# pos 裂解位置，加
        aa_type = "ACDEFGHIKLMNPQRSTVWY"
        SS_type = "HBEGITSC"
        rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                    185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

        with open(dssp_file, "r") as f:
            lines = f.readlines()
        # 提取子序列的dssp feature 加
        # if (pos-10)<0: # 10
        #     s=0
        # else:
        #     s=pos-10
        # e=pos+10
        #########
        seq = ""
        dssp_feature = []

        p = 0
        while lines[p].strip()[0] != "#":
            p += 1
        for i in range(p + 1, len(lines)):
            if lines[i][11]==chain:
                aa = lines[i][13]
                if aa == "!" or aa == "*":
                    continue
                seq += aa
                SS = lines[i][16]
                if SS == " ":
                    SS = "C"
                SS_vec = np.zeros(9) # The last dim represents "Unknown" for missing residues
                SS_vec[SS_type.find(SS)] = 1
                PHI = float(lines[i][103:109].strip())
                PSI = float(lines[i][109:115].strip())
                ACC = float(lines[i][34:38].strip())
                ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
                dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

        return seq, dssp_feature # 子序列的dssp feature seq, dssp_feature

    def match_dssp2(seq, dssp, ref_seq):
        alignments = pairwise2.align.globalxx(ref_seq, seq)
        ref_seq = alignments[0].seqA
        seq = alignments[0].seqB

        SS_vec = np.zeros(9) # The last dim represent "Unknown" for missing residues
        SS_vec[-1] = 1
        padded_item = np.concatenate((np.array([360, 360, 0]), SS_vec))

        new_dssp = []
        for aa in seq: # dssp seq
            if aa == "-":
                new_dssp.append(padded_item)
            else:
                new_dssp.append(dssp.pop(0)) # pop 删除list中第一个元素并返回该值

        matched_dssp = []
        for i in range(len(ref_seq)):
            if ref_seq[i] == "-": # 如果提供的序列中有-，则不append
                continue
            matched_dssp.append(new_dssp[i])

        return matched_dssp

    def transform_dssp2(dssp_feature):
        dssp_feature = np.array(dssp_feature)
        angle = dssp_feature[:,0:2]
        ASA_SS = dssp_feature[:,2:]

        radian = angle * (np.pi / 180)
        dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis = 1)

        return dssp_feature


    def get_dssp2(data_path,dssp_path, ID, ref_seq,chain):
        Uid=ID.split('&')[0].split('_')[-1]# uniprotid
        try:
            if not os.path.exists(dssp_path + ID + ".dssp"):
                # print(Uid,ID)
                os.system("{} -i {} -o {}.dssp".format(DSSP, data_path + '/' + Uid + '.pdb', dssp_path + Uid))

            dssp_seq, dssp_matrix = process_dssp2(dssp_path + Uid + ".dssp",chain)
            # dssp_seq, dssp_matrix = process_dssp(dssp_path + ID + ".dssp")
            # print(dssp_seq,ref_seq)
            if dssp_seq != ref_seq: # 可判断 dssp 解析的seq 和 fasta 输入的seq 是否一致
                # print('序列不一致：',ID,Uid)
                dssp_matrix = match_dssp2(dssp_seq, dssp_matrix, ref_seq)
            np.save(dssp_path + ID + "_dssp.npy", transform_dssp2(dssp_matrix))
            # os.system('rm {}.dssp'.format(dssp_path + ID))
            return 0
        except Exception as e:
            # print(e,Uid)
            return None
    
    
    pdbfasta = {}
    # posDict={}
    with open(fasta_file) as r1: # 输入的是全长序列的fasta文件
        fasta_ori = r1.readlines()
    for i in range(len(fasta_ori)):
        # if fasta_ori[i][0] == ">":
        #     name = fasta_ori[i].split('>')[1].replace('\n','') # 增加 分割name的代码
            
        #     seq = fasta_ori[i+1].replace('\n','')
        #     pdbfasta[name] = seq
        if fasta_ori[i][0] == ">":
            # name = fasta_ori[i].split('>')[1].replace('\n','') # 增加 分割name的代码
            descrL=fasta_ori[i].split('>')[1].replace('\n','').split()
            # # y=int(descrL[1])
            # pos=int(descrL[0].split('&')[1])
            
            # # Uid=descrL[0].split('&')[0].split('_')[-1]# uniprotid
            # Uid=descrL[0].split('&')[0].split('_')[-1]# uniprotid
            # if Uid in error_pdbs: #C14001 'P46934-3','Q99590-2'#C14003 ['Q14789','Q8BTI8','P11881','Q63HN8','Q13315','P25054','P42858','P49792','P50851','Q9UQ35']:
            #     continue
            # else:
                # name=descrL[0]
                # seq = fasta_ori[i+1].replace('\n','')
                # pdbfasta[name] = seq.replace('-','')
                # posDict[name]=pos
            name=descrL[0]
            seq = fasta_ori[i+1].replace('\n','')
            pdbfasta[name] = seq#.replace('X','')
            
    # name=list(pdbfasta.keys())[0]
    # sign = get_dssp2(pdb_path,dssp_path, name ,pdbfasta[name],chain)      
      
    fault_name = []
    for name in pdbfasta.keys():
        
        # sign = get_dssp(pdb_path,dssp_path, name ,pdbfasta[name])
        sign = get_dssp2(pdb_path,dssp_path, name ,pdbfasta[name],chain)# 增加posDict[name] 传pos
        if sign == None:
            fault_name.append(name)
    if fault_name != []:
        np.save(dssp_path+'dssp_fault.npy',fault_name)

def get_dssp_for_pre(fasta_file, pdb_path, dssp_path,protease,chain):
    DSSP = './dssp'


    def process_dssp3(dssp_file,chain):# pos 裂解位置，加
        aa_type = "ACDEFGHIKLMNPQRSTVWY"
        SS_type = "HBEGITSC"
        rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                    185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

        with open(dssp_file, "r") as f:
            lines = f.readlines()
        # 提取子序列的dssp feature 加
        # if (pos-10)<0: # 10
        #     s=0
        # else:
        #     s=pos-10
        # e=pos+10
        #########
        seq = ""
        dssp_feature = []

        p = 0
        while lines[p].strip()[0] != "#":
            p += 1
        for i in range(p + 1, len(lines)):
            if lines[i][11]==chain:
                aa = lines[i][13]
                if aa == "!" or aa == "*":
                    continue
                seq += aa
                SS = lines[i][16]
                if SS == " ":
                    SS = "C"
                SS_vec = np.zeros(9) # The last dim represents "Unknown" for missing residues
                SS_vec[SS_type.find(SS)] = 1
                PHI = float(lines[i][103:109].strip())
                PSI = float(lines[i][109:115].strip())
                ACC = float(lines[i][34:38].strip())
                ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
                dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

        return seq, dssp_feature # 子序列的dssp feature seq, dssp_feature

    def match_dssp3(seq, dssp, ref_seq):
        alignments = pairwise2.align.globalxx(ref_seq, seq)
        ref_seq = alignments[0].seqA
        seq = alignments[0].seqB

        SS_vec = np.zeros(9) # The last dim represent "Unknown" for missing residues
        SS_vec[-1] = 1
        padded_item = np.concatenate((np.array([360, 360, 0]), SS_vec))

        new_dssp = []
        for aa in seq: # dssp seq
            if aa == "-":
                new_dssp.append(padded_item)
            else:
                new_dssp.append(dssp.pop(0)) # pop 删除list中第一个元素并返回该值

        matched_dssp = []
        for i in range(len(ref_seq)):
            if ref_seq[i] == "-": # 如果提供的序列中有-，则不append
                continue
            matched_dssp.append(new_dssp[i])

        return matched_dssp

    def transform_dssp3(dssp_feature):
        dssp_feature = np.array(dssp_feature)
        angle = dssp_feature[:,0:2]
        ASA_SS = dssp_feature[:,2:]

        radian = angle * (np.pi / 180)
        dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis = 1)

        return dssp_feature


    def get_dssp3(data_path,dssp_path, ID, ref_seq,chain):
        Uid=ID.split('&')[0]# uniprotid
        try:
            if not os.path.exists(dssp_path + Uid + ".dssp"):
                # print(Uid,ID)
                os.system("{} -i {} -o {}.dssp".format(DSSP, data_path, dssp_path + Uid))

            dssp_seq, dssp_matrix = process_dssp3(dssp_path + Uid + ".dssp",chain)
            # dssp_seq, dssp_matrix = process_dssp(dssp_path + ID + ".dssp")
            # print(dssp_seq,ref_seq)
            if dssp_seq != ref_seq: # 可判断 dssp 解析的seq 和 fasta 输入的seq 是否一致
                # print('序列不一致：',ID,Uid)
                dssp_matrix = match_dssp3(dssp_seq, dssp_matrix, ref_seq)
            np.save(dssp_path + Uid + "_dssp.npy", transform_dssp3(dssp_matrix))
            # os.system('rm {}.dssp'.format(dssp_path + ID))
            return 0
        except Exception as e:
            # print(e,Uid)
            return None
    
    
    pdbfasta = {}
    # posDict={}
    with open(fasta_file) as r1: # 输入的是全长序列的fasta文件
        fasta_ori = r1.readlines()
    for i in range(len(fasta_ori)):
        # if fasta_ori[i][0] == ">":
        #     name = fasta_ori[i].split('>')[1].replace('\n','') # 增加 分割name的代码
            
        #     seq = fasta_ori[i+1].replace('\n','')
        #     pdbfasta[name] = seq
        if fasta_ori[i][0] == ">":
            # name = fasta_ori[i].split('>')[1].replace('\n','') # 增加 分割name的代码
            descrL=fasta_ori[i].split('>')[1].replace('\n','').split()
            # # y=int(descrL[1])
            # pos=int(descrL[0].split('&')[1])
            
            # # Uid=descrL[0].split('&')[0].split('_')[-1]# uniprotid
            # Uid=descrL[0].split('&')[0].split('_')[-1]# uniprotid
            # if Uid in error_pdbs: #C14001 'P46934-3','Q99590-2'#C14003 ['Q14789','Q8BTI8','P11881','Q63HN8','Q13315','P25054','P42858','P49792','P50851','Q9UQ35']:
            #     continue
            # else:
                # name=descrL[0]
                # seq = fasta_ori[i+1].replace('\n','')
                # pdbfasta[name] = seq.replace('-','')
                # posDict[name]=pos
            name=descrL[0]
            seq = fasta_ori[i+1].replace('\n','')
            pdbfasta[name] = seq#.replace('X','')
            
    name=list(pdbfasta.keys())[0]
    
    sign = get_dssp3(pdb_path,dssp_path, name ,pdbfasta[name],chain)      
    if sign == None:
        print("feature.get_dssp:error!")
    # fault_name = []
    # for name in pdbfasta.keys():
        
    #     # sign = get_dssp(pdb_path,dssp_path, name ,pdbfasta[name])
    #     sign = get_dssp2(pdb_path,dssp_path, name ,pdbfasta[name],chain)# 增加posDict[name] 传pos
    #     if sign == None:
    #         fault_name.append(name)
    # if fault_name != []:
    #     np.save(dssp_path+'dssp_fault.npy',fault_name)

def get_dssp_feature(fasta_file, pdb_path, dssp_path,protease,chain,args):
    temp_path_naccess=args.outputpath+'/naccess'
    os.makedirs(temp_path_naccess, exist_ok=True)
    DSSP = './dssp'
    pdb_parser = PDB.PDBParser(QUIET=True)
    HYDROGEN_BOND_DONORS: Dict[str, Dict[str, int]] = {
    "ARG": {"NE": 1, "NH1": 2, "NH2": 2},
    "ASN": {"ND2": 2},
    "GLN": {"NE2": 2},
    "HIS": {"ND1": 2, "NE2": 2},
    "LYS": {"NZ": 3},
    "SER": {"OG": 1},
    "THR": {"OG1": 1},
    "TYR": {"OH": 1},
    "TRP": {"NE1": 1},
    }

    HYDROGEN_BOND_ACCEPTORS: Dict[str, Dict[str, int]] = {
        "ASN": {"OD1": 2},
        "ASP": {"OD1": 2, "OD2": 2},
        "GLN": {"OE1": 2},
        "GLU": {"OE1": 2, "OE2": 2},
        "HIS": {"ND1": 1, "NE2": 1},
        "SER": {"OG": 2},
        "THR": {"OG1": 2},
        "TYR": {"OH": 1},
    }
    
    def parse_naccess_output(d1):
        accessibility_data = {
            'all_atoms': [],
            'total_side_chain': [],
            'main_chain': [],
            'non_polar_side_chain': [],
            'all_polar_side_chain': [],
        }

        for line in d1:
            if line.startswith('RES') and 'ALL' not in line:
                columns = line.split()
                if len(columns)==14:
                    accessibility_data['all_atoms'].append(float(columns[5]))
                    accessibility_data['total_side_chain'].append( float(columns[7]))
                    accessibility_data['main_chain'].append(float(columns[9]))
                    accessibility_data['non_polar_side_chain'].append(float(columns[11]))
                    accessibility_data['all_polar_side_chain'].append(float(columns[13]))
                elif len(columns)==13:
                    accessibility_data['all_atoms'].append(float(columns[4]))
                    accessibility_data['total_side_chain'].append( float(columns[6]))
                    accessibility_data['main_chain'].append(float(columns[8]))
                    accessibility_data['non_polar_side_chain'].append(float(columns[10]))
                    accessibility_data['all_polar_side_chain'].append(float(columns[12]))

        return accessibility_data
    
    def process_dssp(dssp_file,chain):
        aa_type = "ACDEFGHIKLMNPQRSTVWY"
        SS_type = "HBEGITSC"
        rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                    185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

        with open(dssp_file, "r") as f:
            lines = f.readlines()
        seq = ""
        dssp_feature = []

        p = 0
        while lines[p].strip()[0] != "#":
            p += 1
        for i in range(p + 1, len(lines)):
            if lines[i][11]==chain:
                aa = lines[i][13]
                if aa == "!" or aa == "*":
                    continue
                seq += aa
                SS = lines[i][16]
                if SS == " ":
                    SS = "C"
                SS_vec = np.zeros(9) 
                SS_vec[SS_type.find(SS)] = 1
                PHI = float(lines[i][103:109].strip())
                PSI = float(lines[i][109:115].strip())
                ACC = float(lines[i][34:38].strip())
                ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
                
                res=protein_letters_1to3[aa].upper()
                if res not in HYDROGEN_BOND_DONORS.keys():
                    HBD = 0
                else:
                    HBD = sum(HYDROGEN_BOND_DONORS[res].values())
                HBD = np.array([HBD]).astype(int)
                if res not in HYDROGEN_BOND_ACCEPTORS.keys():
                    HBC = 0
                else:
                    HBC = sum(HYDROGEN_BOND_ACCEPTORS[res].values())
                HBC = np.array([HBC]).astype(int)
                dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]),HBD,HBC, SS_vec)))

        return seq, dssp_feature 

    def match_dssp(seq, dssp, ref_seq):
        alignments = pairwise2.align.globalxx(ref_seq, seq)
        ref_seq = alignments[0].seqA
        seq = alignments[0].seqB

        SS_vec = np.zeros(9) 
        SS_vec[-1] = 1
        padded_item = np.concatenate((np.array([360, 360, 0,0,0]), SS_vec))

        new_dssp = []
        for aa in seq: # dssp seq
            if aa == "-":
                new_dssp.append(padded_item)
            else:
                new_dssp.append(dssp.pop(0)) 

        matched_dssp = []
        for i in range(len(ref_seq)):
            if ref_seq[i] == "-":
                continue
            matched_dssp.append(new_dssp[i])

        return matched_dssp

    def transform_dssp(dssp_feature):
        dssp_feature = np.array(dssp_feature)
        angle = dssp_feature[:,0:2]
        ASA_HB_SS = dssp_feature[:,2:]

        radian = angle * (np.pi / 180)
        dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_HB_SS], axis = 1)

        return dssp_feature

    def get_dssp(data_path,dssp_path, ID, ref_seq,chain):

        if not os.path.exists(data_path):
            print('PDB file not exist!',data_path)
        try:
            if not os.path.exists(dssp_path + ID + ".dssp"):
                os.system("{} -i {} -o {}.dssp".format(DSSP, data_path, dssp_path + ID))

            dssp_seq, dssp_matrix = process_dssp(dssp_path + ID + ".dssp",chain)
            # dssp_seq, dssp_matrix = process_dssp(dssp_path + ID + ".dssp")
            # print(dssp_seq,ref_seq)
            if dssp_seq != ref_seq: 
                print('序列不一致：',ID)
                dssp_matrix = match_dssp(dssp_seq, dssp_matrix, ref_seq)
            np.save(dssp_path + ID + "_dssp_crf.npy", transform_dssp(dssp_matrix))
            # os.system('rm {}.dssp'.format(dssp_path + ID))
            return 0
        except Exception as e:
            print(e,Uid)
            return None
    pdbfasta = {}
    posDict={}
    with open(fasta_file) as r1:
        fasta_ori = r1.readlines()
    for i in range(len(fasta_ori)):

        if fasta_ori[i][0] == ">":
            descrL=fasta_ori[i].split('>')[1].replace('\n','').split()
            Uid=descrL[0].split('&')[0].split('_')[-1]
            if Uid in []:
                continue
            else:
                name=descrL[0]
                seq = fasta_ori[i+1].replace('\n','')
                pdbfasta[name] = seq.replace('X','')

    fault_name = []
    for name in pdbfasta.keys():
        Uid=name.split('&')[0].split('_')[-1]
        if not os.path.exists(dssp_path + Uid + "_dssp_crf.npy"):
            sign = get_dssp(pdb_path+'/'+str(Uid)+'.pdb',dssp_path, Uid ,pdbfasta[name],chain)
            if sign == None:
                fault_name.append(name)
        if not os.path.exists(dssp_path + Uid + "_naccess.pkl"):   
            # print('naccess',Uid)     
            structure= pdb_parser.get_structure('protein', pdb_path+'/'+str(Uid)+'.pdb')
            d1, d2 = NACCESS.run_naccess(structure[0],pdb_path+'/'+str(Uid)+'.pdb', naccess="../naccess/naccess",temp_path=temp_path_naccess)#"code_test_result/naccess"
            naccess_data = parse_naccess_output(d1)
            with open(dssp_path + Uid + "_naccess.pkl",'wb') as file:
                pickle.dump(naccess_data,file)
            # np.save(dssp_path + Uid + "_naccess.npy", naccess_data)
            os.system('rm -r {}/tmp*'.format(temp_path_naccess))
    if fault_name != []:
        np.save(dssp_path+'dssp_fault.npy',fault_name)
    
def get_other_feature(fasta_file, pdb_path, output_path,protease,chain):

    def calculate_contact_number(structure,chain_id,pos, threshold=10.0):
        contact_number = 0
        if (pos-4)<0: # 10
            s=0
        else:
            s=pos-4
        e=pos+4
        # for model in structure:
        for chain in structure[0]:
            if chain.get_id()==chain_id: # 只计算目标链
                #只计算P4'至P4位置氨基酸
                chain=list(chain)
                for residue1 in chain[s:e]:
                    for residue2 in chain[s:e]:
                        if residue1.id[0] == ' ' and residue2.id[0] == ' ' and residue1 != residue2:
                            for atom1 in residue1:
                                for atom2 in residue2:
                                    distance = atom1 - atom2
                                    if distance < threshold:
                                        contact_number += 1
                                        break
        return contact_number

    def calculate_half_sphere_atoms(structure,atom_name,chain_id,pos):
        upper_half_sphere = 0
        lower_half_sphere = 0
        if (pos-4)<0: # 10
            s=0
        else:
            s=pos-4
        e=pos+4
        # for model in structure:
        for chain in structure[0]:
            if chain.get_id()==chain_id:# 只计算目标链
                chain=list(chain)
                for residue in chain[s:e]:
                    for atom in residue:
                        # print(atom.id)
                        if atom.id==atom_name:
                            coords = atom.get_coord()
                            if coords[2] >= 0:
                                upper_half_sphere += 1
                            else:
                                lower_half_sphere += 1

        return upper_half_sphere, lower_half_sphere

    def calculate_packing(structure,chain_id,pos):
        packing_scores = []
        if (pos-4)<0: # 10
            s=0
        else:
            s=pos-4
        e=pos+4
        # for model in structure:
        for chain in structure[0]:
            if chain.get_id()==chain_id:
                chain=list(chain)
                for residue1 in chain[s:e]:
                    for residue2 in chain[s:e]:
                        if residue1.id[0] == ' ' and residue2.id[0] == ' ' and residue1 != residue2:
                            for atom1 in residue1:
                                for atom2 in residue2:
                                    distance = atom1 - atom2
                                    packing_scores.append(distance)

        return np.mean(packing_scores)#packing_scores#
    def calculate_protrusion_and_depth_index(structure,chain_id,pos):
        # parser = PDBParser(QUIET=True)
        # structure = parser.get_structure('protein', pdb_file)
        if (pos-4)<0: # 10
            s=0
        else:
            s=pos-4
        e=pos+4
        # 初始化变量
        protrusion_index = []
        depth_index = []

        # for model in structure:
        for chain in structure[0]:
            if chain.get_id()==chain_id:
                chain=list(chain)[s:e]
                for i, residue in enumerate(chain):
                    # 获取残基坐标
                    current_coords = residue['CA'].get_coord()
                    # 计算前一个和后一个残基的坐标
                    if i > 0:
                        try:
                            prev_coords = chain[i - 1]['CA'].get_coord()
                        except:
                            prev_coords=current_coords
                    else:
                        prev_coords = current_coords

                    if i < len(chain) - 1:
                        try:
                            next_coords = chain[i + 1]['CA'].get_coord()
                        except:
                            next_coords =current_coords
                    else:
                        next_coords = current_coords

                    # 计算Protrusion和Depth Index
                    protrusion = np.linalg.norm(np.cross(next_coords - current_coords, current_coords - prev_coords))
                    depth = np.dot(next_coords - prev_coords, current_coords - prev_coords)

                    protrusion_index.append(protrusion)
                    depth_index.append(depth)

        return protrusion_index, depth_index
    
    def calculate_b_factor_statistics(structure,chain_id,pos):

        b_factors = []
        if (pos-4)<0: # 10
            s=0
        else:
            s=pos-4
        e=pos+4
        # for model in structure[0]:
        for chain in structure[0]:
            if chain.get_id()==chain_id:
                chain=list(chain)[s:e]
                for residue in chain:
                    for atom in residue:
                        # 获取B-factor值
                        b_factor = atom.get_bfactor()
                        # 将B-factor添加到列表中
                        b_factors.append(b_factor)

        # 计算平均值和标准差
        # print(b_factors)
        mean_b_factor = np.mean(b_factors)
        # std_dev_b_factor = np.std(b_factors)

        return mean_b_factor#, std_dev_b_factor
    
    pdbfasta = {}
    posDict={}
    with open(fasta_file) as r1:
        fasta_ori = r1.readlines()
    for i in range(len(fasta_ori)):

        if fasta_ori[i][0] == ">":
            # name = fasta_ori[i].split('>')[1].replace('\n','') # 增加 分割name的代码
            descrL=fasta_ori[i].split('>')[1].replace('\n','').split()
            # y=int(descrL[1])
            pos=int(descrL[0].split('&')[1])

            Uid=descrL[0].split('&')[0].split('_')[-1]# uniprotid
            if Uid in []:#error_pdbs+['Q8IVL0','P12270','Q13813','P12270','Q13813','P01514']:
                continue
            else:
                name=descrL[0]
                # seq = fasta_ori[i+1].replace('\n','')
                # pdbfasta[name] = seq.replace('X','')
                posDict[name]=pos
    pdb_parser = PDB.PDBParser(QUIET=True)
    
    for name in posDict.keys():
        
        
        if not os.path.exists(output_path + name + "_other.npy"):
            Uid=name.split('&')[0].split('_')[-1]
            structure= pdb_parser.get_structure('protein', pdb_path+'/'+str(Uid)+'.pdb')
            if not os.path.exists(pdb_path+'/'+str(Uid)+'.pdb'):
                print('PDB file not exist!',pdb_path+'/'+str(Uid)+'.pdb')
            # 计算联系数
            contact_num = calculate_contact_number(structure,chain,posDict[name])
            # print(f"Contact Number: {contact_num}")
            # 计算Ca和Cb原子的数量
            # print(name)
            upper_ca, lower_ca = calculate_half_sphere_atoms(structure, 'CA',chain,posDict[name])
            upper_cb, lower_cb = calculate_half_sphere_atoms(structure, 'CB',chain,posDict[name])
            packing_score = calculate_packing(structure,chain,posDict[name])
            protrusion_values, depth_values = calculate_protrusion_and_depth_index(structure,chain,posDict[name])
            mean_b_factor= calculate_b_factor_statistics(structure,chain,posDict[name])
            Other_features=np.array([contact_num, upper_ca, lower_ca, upper_cb, lower_cb,packing_score,mean_b_factor]+protrusion_values+depth_values)
            np.save(output_path + name + "_other.npy", Other_features)

def structure_features(dataset, index, fasta_file,chain,feature_path,pdb_path,protease,args,device="cpu"):
    IDs = list(dataset.keys())
    # 加，提取子序列的特征
    posDict={}
    yDict={}
    for key,value in dataset.items():
        posDict[key]=value[0]
        yDict[key]=value[1]
    data=[]
    y=[]
    new_IDs=[]
    for name in IDs:
        check_point=0
        pdbid=name.split('&')[0].split('_')[-1]# uniprotid
        # pdbid=name.split('&')[0]
        # 提取子序列的dssp feature 加
        pos=posDict[name]
        
        if (pos-4)<0:
            s=0
        else:
            s=pos-4
        e=pos+4
        #########
        with torch.no_grad():
            if not os.path.exists(feature_path + pdbid + "_dssp_crf.npy") or not os.path.exists(feature_path + pdbid + "_naccess.pkl"):# 计算的是全长序列对应的dssp 特征，下面需要提取子序列的dssp 特征
                # print('run_get_dssp')
                # features.get_dssp(self.fasta_file, self.output_esmfold, self.output_dssp)
                get_dssp_feature(fasta_file, pdb_path, feature_path,protease,chain,args)
            if not os.path.exists(feature_path + name + "_other.npy"):
                # features.get_dssp(self.fasta_file, self.output_esmfold, self.output_dssp)
                get_other_feature(fasta_file, pdb_path,feature_path,protease,chain)
            with open(feature_path + pdbid + "_naccess.pkl", 'rb') as file:
                naccess = pickle.load(file)
            # naccess = torch.tensor(np.load(feature_path + pdbid + "_naccess.npy"))
            dssp = np.load(feature_path + pdbid + "_dssp_crf.npy")
            _other = np.load(feature_path + name + "_other.npy")
            dssp=dssp[s:e]
            # print(pos,dssp.shape,_other.shape)
            for k in naccess.keys():
                if len(naccess[k][s:e])==0:
                    # print("naccess feature empty",name)
                    check_point=1
            if check_point==1:
                continue
            naccess_cat=np.concatenate([np.array(naccess[k][s:e]) for k in naccess.keys()], axis=0)
            if int(dssp.flatten().shape[0])!=128 or int(_other.shape[0])!=23 or int(naccess_cat.shape[0])!=40:
                continue
            # print(dssp.view(-1).shape,_other.shape,naccess_cat.shape)
            data.append(np.concatenate([dssp.flatten(), _other,naccess_cat], axis=0))
            y.append(yDict[name])
            new_IDs.append(name)
    return np.vstack(data),np.array(y),new_IDs
        

    
    
   
