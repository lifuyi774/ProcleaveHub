import os,re,sys,shutil
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"  
import torch
import pickle
import argparse
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from data import *
from model import *
import torch.optim as optim
from torch_geometric.data import DataLoader
# from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from evaluation import evaluate_family_classifier
from sklearn.metrics import accuracy_score 
from features import structure_features
from Bio import SeqIO
from sklearn.metrics import matthews_corrcoef,precision_score,f1_score,recall_score
# from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
from utils import *
torch.set_num_threads(20)
torch.manual_seed(1234)
import warnings
warnings.filterwarnings("ignore")



def train_epoch(model, train_dataloader,optimizer,loss_function,graphType,train_loader1=None,train_loader2=None):
    # print('Training...')
    model.train()
    losses = []
    preds_mapper=[{'0':0,'1':1}]
    results = {}
    y_true_list = []
    if graphType=='ensemble':
        dataloader_iterator1 = iter(train_loader1)
        dataloader_iterator2 = iter(train_loader2)
    
    for i,data in enumerate(train_dataloader):

        if graphType=='ensemble':
            try:
                data1 = next(dataloader_iterator1)
                data2 = next(dataloader_iterator2)
            except StopIteration:
                dataloader_iterator1 = iter(train_loader1)
                data1 = next(dataloader_iterator1)
                dataloader_iterator2 = iter(train_loader2)
                data2 = next(dataloader_iterator2)
            data = data.to(args.device)
            data1 = data1.to(args.device)
            data2 = data2.to(args.device)
            target=data.y
            out1 = model(data,data1,data2)
        elif graphType=='PBMLP':
            seq_pro,seq_ind,target = data
            seq_pro,seq_ind,target=seq_pro.to(args.device),seq_ind.to(args.device),target.to(args.device)
            target=target.squeeze()
            out1 = model(seq_pro,seq_ind)    
        else:
            data = data.to(args.device)
            target=data.y
            out1 = model(data) 

        loss = 0

        # loss
        loss = loss_function(out1[0], target.long())
        losses.append(loss.item())
        #print(f'single train loss: {loss.item()}')
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()


        predictions = [
        torch.argmax(torch.nn.Softmax(dim=1)(task_output), dim=1)
        .cpu()
        .detach()
        .numpy()
        .reshape(-1)
        for task_output in out1
        ]
        for task_idx in range(1):
            results[task_idx] = results.get(task_idx, []) + [predictions[task_idx]] 

        y_true_list += list(target)

        
    results = [np.hstack(task_res) for task_res in results.values()]
    # map all task individually
    results = [
        np.vectorize(preds_mapper[task_idx].get)(task_res.astype(str))
        for task_idx, task_res in enumerate(results)
    ]
    y_true_tensor2 = torch.stack(y_true_list)
    y_true=y_true_tensor2.cpu().numpy()
    yT=y_trueTOnumpy(y_true)
    y_prdclass=preClass(results)
    train_loss = np.mean(losses)
    accs=[accuracy_score(yT[:,i],y_prdclass[:,i]) for i in range(1)]

    
    return train_loss,accs[0]

def train(n_epochs,model,optimizer,loss_function,train_dataloader,
          test_dataloader,model_dir,results_dir,graphType,train_dataloader1=None,
          test_dataloader1=None,train_dataloader2=None,test_dataloader2=None):
    
    # train_losses = []
    val_losses = []
    # val_accuracies = []
    # val_mccs = []

    best_acc=0
    best_mcc=0
    loop = tqdm(range(n_epochs), total =n_epochs)
    # for epoch in range(n_epochs):
    for epoch in loop:
        train_loss,train_acc = train_epoch(model, train_dataloader,optimizer,loss_function,
                                           graphType,train_loader1=train_dataloader1,
                                           train_loader2=train_dataloader2) 
        val_loss,valid_acc,valid_mcc = evaluate_family_classifier(
            model,test_dataloader,loss_function,graphType,mode='val',
            device=args.device, batch_size=args.batch_size,
            final=False,dir=results_dir,test_loader1=test_dataloader1,test_loader2=test_dataloader2)
        if epoch ==0:
            torch.save(model,  model_dir + 'model_{}.pth'.format(graphType))
        # early_stopping 
        if len(val_losses)!=0 and min(val_losses) > val_loss:
            val_losses.clear() 
            # val_mccs.clear()
            # val_accuracies.clear()
            val_losses.append(val_loss)
            # val_mccs.append(valid_mcc)
            # val_accuracies.append(valid_acc)
            best_acc=valid_acc
            best_mcc=valid_mcc
            # torch.save(model.state_dict(), model_dir + 'model_{}.pt'.format(graphType))
            torch.save(model,  model_dir + 'model_{}.pth'.format(graphType))
        else:
            val_losses.append(val_loss)
        if len(val_losses) > args.early_stopping and min(val_losses) < val_loss:
            print('\n')
            print("Training terminated because of early stopping")
            print("Best val_loss: {}".format(round(min(val_losses),6)))
            print("Best val_acc: {}".format(round(best_acc,3)))
            print("Best val_mcc: {}".format(round(best_mcc,3)))
            break        


        loop.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
        loop.set_postfix(train_loss = train_loss,train_acc = train_acc,valid_loss=val_loss,valid_acc=valid_acc)
    


    return best_acc,best_mcc

def Predict(model,test_loader,graphType='knn',device='cpu',test_loader1=None,test_loader2=None): 
    model=model.to(device)
    model.eval()

    with torch.no_grad():
        trainresults=[]
        if graphType=='ensemble':
            dataloader_iterator1 = iter(test_loader1)
            dataloader_iterator2 = iter(test_loader2)
        for i, data in enumerate(test_loader):
            if graphType=='ensemble':
                try:
                    data1 = next(dataloader_iterator1)
                    data2 = next(dataloader_iterator2)
                except StopIteration:
                    dataloader_iterator1 = iter(test_loader1)
                    data1 = next(dataloader_iterator1)
                    dataloader_iterator2 = iter(test_loader2)
                    data2 = next(dataloader_iterator2)
                data = data.to(device)
                data1 = data1.to(device)
                data2 = data2.to(device)
                # target=data.y
                out1 = model(data,data1,data2)
            elif graphType=='PBMLP':
                seq_pro,seq_ind,_= data
                seq_pro,seq_ind=seq_pro.to(device),seq_ind.to(device)
                out1 = model(seq_pro,seq_ind) 
            else:
                data = data.to(device)
                # target=data.y
                out1 = model(data)
            y_pred = torch.nn.Softmax(dim=1)(out1[0]).cpu().detach().numpy()
            # print('y_pre',y_pred)
            trainresults.append(y_pred)
        scores= np.vstack(trainresults)
        return scores

def read_inputfiles(mode,inputfile,args,inputType=None,crf=None,chain='A',dataType='train'):

    protease=args.protease
    if mode =='prediction':
        test_data={}
        if inputType =='pdb':
            filename = os.path.basename(inputfile)
            # print(filename)
            if not filename.endswith(".pdb"):
                print('Please check the input file, which should be in pdb format(.pdb)')
                sys.exit(1)
            if crf!=None:
                tmp_fasta_file=args.outputpath+'/temp_crf.fasta'
            else:
                tmp_fasta_file=args.outputpath+'/temp.fasta'
            
            f=open(tmp_fasta_file,'w')
            
            # chain = {record.id: record.seq for record in SeqIO.parse(pdb_file, 'pdb-atom')}
            chainSeq = {str(record.id).split(':')[-1]: str(record.seq) for record in SeqIO.parse(inputfile, 'pdb-atom')}
            # pdbid, ext = os.path.splitext(filename)
            pdbid ='.'.join(filename.split('.')[:-1])
            seq=chainSeq[chain]
            seq=re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(seq).upper())
            if crf==None: 
                f.write('>'+str(pdbid)+'\n')
                f.write(str(seq)+'\n')
            
            
            if len(seq) >= 20:
                for j in range(0, len(seq) - 20 + 1):

                    if crf !=None:
                        f.write('>'+str(pdbid)+'&'+str(j+10)+' '+'1'+'\n')
                        f.write(str(seq)+'\n')
                    name=str(pdbid)+'&'+str(j+10)
                    test_data[name]=(int(j+10),1,chainSeq[chain].replace('-',''))
            else:

                if crf !=None: 
                    f.write('>'+str(pdbid)+'&'+str(int(len(seq)/2))+' '+'1'+'\n')
                    f.write(str(seq)+'\n')
                name=str(pdbid)+'&'+str(int(len(seq)/2))
                test_data[name]=(int(len(seq)/2),1,chainSeq[chain].replace('-',''))  
            f.close()
            return tmp_fasta_file,test_data
        elif inputType =='fasta':
            
            for record in SeqIO.parse(inputfile,'fasta'):
                seq=str(record.seq)
                id=str(record.id)
                if len(seq) >= 20:
                    for j in range(0, len(seq) - 20 + 1):

                        name=str(id)+'&'+str(j+10)
                        test_data[name]=(int(j+10),1,chainSeq[chain].replace('-',''))
                else:

                    name=str(id)+'&'+str(int(len(seq)/2))
                    test_data[name]=(int(len(seq)/2),1,chainSeq[chain].replace('-',''))
        return inputfile,test_data
    elif mode=='train':

        data={}
        tmp_fasta_file=args.outputpath+'/'+str(dataType)+'.fasta'
        f=open(tmp_fasta_file,'w')
        with open(inputfile) as r1:
            lines = r1.readlines()

        for line in lines:
            if '\t' in line:
                lineList=line.strip().split('\t')
            elif ',' in line:
                lineList=line.strip().split(',')
            else:
                lineList=line.strip().split()
            # lineList=line.strip().split('\t')
            id_=lineList[2]
            if id_ in []:
                continue
            label=int(lineList[0])
            pos=int(lineList[1])
            chainType=lineList[3]
            # chain = {record.id: record.seq for record in SeqIO.parse(pdb_file, 'pdb-atom')}
            pdbpath=args.pdb_path+'/'+str(id_)+'.pdb' 
            if not os.path.exists(pdbpath):
                print(f'File not found:{pdbpath}')
                sys.exit(1)
            try:
                chainSeq = {str(record.id).split(':')[-1]: str(record.seq) for record in SeqIO.parse(pdbpath, 'pdb-atom')}
            except:
                print(f'Error in reading pdb file! Please check {str(id_)}.pdb ')
                sys.exit(1)
            # pdbid, ext = os.path.splitext(filename)
            if chainType not in list(chainSeq): 
                print(f'Error chaintype:{line}')
                continue
            seq=chainSeq[chainType]
            seq=re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(seq).upper())
            name=str(protease)+'_'+str(id_)+'&'+str(pos)
            data[name] = (pos,label,seq)
            f.write('>'+name+' '+str(label)+'\n')
            f.write(str(seq)+'\n')
        f.close()
        return tmp_fasta_file,data

def test(Gtype,model_dir,results_dir,loss_function,test_dataloader,test_dataloader1=None,test_dataloader2=None):
    print('\n')
    print(f'*****************The best model is {Gtype}-based model. Testing with {Gtype}-based model.*****************')
    if Gtype=='ensemble':
        # model =GVPEnsemble(1044, 128, 5, 0.95, 0.1).to(args.device)
        # # print('*****{}-based model*****\n'.format(Gtype),model)
        # # load model
        # checkpoint = torch.load(model_dir + 'model_{}.pt'.format(Gtype))
        # model.load_state_dict(checkpoint)
        model = torch.load(model_dir + 'model_{}.pth'.format(Gtype))
        model = model.to(args.device)
        acc,mcc,Recall,F1,Precision,se,sp,auc= evaluate_family_classifier(
            model,test_dataloader,loss_function,Gtype,mode='test',
            device=args.device, batch_size=args.batch_size,
            final=True,dir=results_dir,test_loader1=test_dataloader1,test_loader2=test_dataloader2)     
    else:
        # model =GLM(1044, 128, 5, 0.95, 0.1).to(args.device)
        # # print('*****{}-based model*****\n'.format(Gtype),model)
        # checkpoint = torch.load(model_dir + 'model_{}.pt'.format(Gtype))
        # model.load_state_dict(checkpoint)
        model = torch.load(model_dir + 'model_{}.pth'.format(Gtype))
        model = model.to(args.device)
        acc,mcc,Recall,F1,Precision,se,sp,auc = evaluate_family_classifier(
            model,test_dataloader,loss_function,Gtype,mode='test',
            device=args.device, batch_size=args.batch_size,
            final=True,dir=results_dir)
    resultDF = pd.DataFrame()
    resultDF['Model']=[Gtype]
    resultDF['Protease']=[args.protease]
    resultDF['ACC']=[acc]
    resultDF['MCC']=[mcc]
    resultDF['Recall']=[Recall]
    resultDF['F1']=[F1]
    resultDF['Precision']=[Precision]
    resultDF['Sensitivity']=[se]
    resultDF['Specificity']=[sp]
    resultDF['AUC']=[auc]
    resultDF.to_csv(args.outputpath + '/performance.csv', index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='prediction',required=True,help='Three modes can be used: prediction, TrainYourModel, UseYourOwnModel, only select one mode each time.')
    parser.add_argument('--protease', type=str, required=True,help='The protease you want to predict cleavage to, eg: A01.001' + '\n'
                            'Or if you want to build a new model, please create a name. There should no space in the model name.')
    parser.add_argument('--chain', type=str, default='A',help='A or B')
    parser.add_argument('--outputpath', type=str,required=True,default='results', help='The path of output.')
    parser.add_argument('--inputpath', type=str, help='The path of the training set file.  Each entry in the training set contains the label, cleavage site position, pdb_id (uniprot accession), and chain type, which are separated by commas.')
    parser.add_argument('--test_file', type=str, help='The path of the test set file.  Each entry in the training set contains the label, cleavage site position, pdb_id (uniprot accession), and chain type, which are separated by commas.')
    parser.add_argument('--pre_file', type=str, default=None,help='The path of the pdb file to predict.')
    parser.add_argument('--model_file', type=str, default=None,help='The path of the model file to deploy.')
    parser.add_argument('--pdb_path', type=str, default=None,help='The path of the pdb files for the training (test or validation) set.')
    parser.add_argument('--inputType', type=str, default='pdb',help='pdb')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--device', default="cpu", help='cpu or cuda')
    parser.add_argument('--n_epochs', type=int, default=200, help='The number of training epoch.')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate.')
    parser.add_argument('--early_stopping', type=int, default=10, help='Number of epochs for early stopping.')
    args = parser.parse_args()
    args.dataset_path=args.outputpath+'/proteases_structure_data/'
    args.feature_path=args.outputpath+'/proteases_prottrans/'
    args.output_prottrans=args.outputpath+'/proteases_prottrans/'
    args.output_esmfold=args.outputpath+'/proteases_structure_data/'
    args.output_dssp=args.outputpath+'/proteases_structure_data/'
    os.makedirs(args.output_prottrans, exist_ok=True)
    os.makedirs(args.dataset_path, exist_ok=True)
    os.makedirs(args.feature_path, exist_ok=True)
    
    
    if args.mode=="prediction":
        proteaeList = str(args.protease).strip().split(',')
        if len(proteaeList)==0:
            print('Please select at least one protease!')
            sys.exit(1)
        if args.inputType!='pdb':
            print('Please provide the correct inputType!')
            sys.exit(1)
        checkpoint_pre = 0
        checkpoint_knn=0
        checkpoint_PBMLP=0
        checkpoint_rball=0
        checkpoint_sequential=0
        checkpoint_crf=0
        pbml_proteases=list(np.load('../models/info_/pbml_proteases.npy'))
        ensemble_proteases=list(np.load('../models/info_/ensemble_proteases.npy'))
        knn_proteases=list(np.load('../models/info_/knn_proteases.npy'))
        sequential_proteases=list(np.load('../models/info_/sequential_proteases.npy'))
        crf_proteases=list(np.load('../models/info_/crf_proteases.npy'))
        rball_proteases=list(np.load('../models/info_/rball_proteases.npy'))
        tmp_fasta_file,test_data=read_inputfiles('prediction',args.pre_file,args,args.inputType,crf=None,chain=args.chain)
        for proteasesOne in proteaeList:            
            if proteasesOne in pbml_proteases: 
                if checkpoint_PBMLP==0:
                    test_dataset_PBMLP = ProteinSequenceDataset1(test_data, range(len(test_data)), tmp_fasta_file,args,graph_type='PBMLP')
                    test_dataloader_PBMLP = DataLoader(test_dataset_PBMLP, batch_size = 1, shuffle=False, drop_last=False, prefetch_factor=2)
                try:
                    model = torch.load('../models/{}/model_PBMLP.pth'.format(proteasesOne))
                    pre_scores=Predict(model,test_dataloader_PBMLP,graphType='PBMLP')
                except:
                    print(f'The model of {proteasesOne} failed to load, please select another type of protease!')
                    sys.exit(1)
                model_type='PBMLP'
                checkpoint_PBMLP=1
            elif proteasesOne in knn_proteases: # knn
                if checkpoint_knn==0:
                    test_dataset_knn  = ProteinGraphDataset2(test_data, range(len(test_data)), tmp_fasta_file,args,graph_type='knn')
                    test_dataloader_knn = DataLoader(test_dataset_knn, batch_size = 1, shuffle=False, drop_last=False, prefetch_factor=2)
                try:
                    model = torch.load('../models/{}/model_knn_1.pth'.format(proteasesOne))
                    pre_scores=Predict(model,test_dataloader_knn,graphType='knn')
                except:
                    print(f'The model of {proteasesOne} failed to load, please select another type of protease!')
                    sys.exit(1)
                model_type='knn'
                checkpoint_knn=1
            elif proteasesOne in rball_proteases: # rball
                if checkpoint_rball==0:
                    test_dataset_rball  = ProteinGraphDataset2(test_data, range(len(test_data)), tmp_fasta_file,args,graph_type='rball')
                    test_dataloader_rball = DataLoader(test_dataset_rball, batch_size = 1, shuffle=False, drop_last=False, prefetch_factor=2)
                try:
                    model = torch.load('../models/{}/model_rball_1.pth'.format(proteasesOne))
                    pre_scores=Predict(model,test_dataloader_rball,graphType='rball')
                except:
                    print(f'The model of {proteasesOne} failed to load, please select another type of protease!')
                    sys.exit(1)    
                model_type='rball'
                checkpoint_rball=1
            elif proteasesOne in sequential_proteases: # sequential
                if checkpoint_sequential==0:
                    test_dataset_sequential  = ProteinGraphDataset2(test_data, range(len(test_data)), tmp_fasta_file,args,graph_type='sequential')
                    test_dataloader_sequential = DataLoader(test_dataset_sequential, batch_size = 1, shuffle=False, drop_last=False, prefetch_factor=2)
                try:
                    model = torch.load('../models/{}/model_sequential_1.pth'.format(proteasesOne))
                    pre_scores=Predict(model,test_dataloader_sequential,graphType='sequential')  
                except:
                    print(f'The model of {proteasesOne} failed to load, please select another type of protease!')
                    sys.exit(1)
                model_type='sequential'
                checkpoint_sequential=1
            elif proteasesOne in ensemble_proteases: # ensemble

                if checkpoint_knn==0:
                    test_dataset_knn  = ProteinGraphDataset2(test_data, range(len(test_data)), tmp_fasta_file,args,graph_type='knn')
                    test_dataloader_knn = DataLoader(test_dataset_knn, batch_size = 1, shuffle=False, drop_last=False, prefetch_factor=2)
                if checkpoint_rball==0:
                    test_dataset_rball  = ProteinGraphDataset2(test_data, range(len(test_data)), tmp_fasta_file,args,graph_type='rball')
                    test_dataloader_rball = DataLoader(test_dataset_rball, batch_size = 1, shuffle=False, drop_last=False, prefetch_factor=2)
                if checkpoint_sequential==0:
                    test_dataset_sequential  = ProteinGraphDataset2(test_data, range(len(test_data)), tmp_fasta_file,args,graph_type='sequential')
                    test_dataloader_sequential = DataLoader(test_dataset_sequential, batch_size = 1, shuffle=False, drop_last=False, prefetch_factor=2)
                try:
                    model = torch.load('../models/{}/model_ensemble_1.pth'.format(proteasesOne))
                    pre_scores=Predict(model,test_dataloader_sequential,graphType='ensemble',test_loader1=test_dataloader_knn,test_loader2=test_dataloader_rball)
                except:
                    print(f'The model of {proteasesOne} failed to load, please select another type of protease!')
                    sys.exit(1)
                model_type='ensemble'
                checkpoint_rball=1
                checkpoint_knn=1
                checkpoint_sequential=1
            elif proteasesOne in crf_proteases:
                if checkpoint_crf==0:
                    tmp_fasta_file_crf,test_data=read_inputfiles('prediction',args.pre_file,args,args.inputType,crf=True,chain=args.chain)
                    pdbpath=os.path.dirname(args.pre_file)
                    test_dataset,test_y,new_ids=structure_features(test_data,range(len(test_data)),tmp_fasta_file_crf,args.chain,args.dataset_path,pdbpath, proteasesOne,args,True)
                    lowess = sm.nonparametric.lowess
                    test_dataset=MyLOWESS(test_dataset,lowess)
                    X_test_feats = [featuresProcess(sentence) for sentence in test_dataset]
                try:
                    with open('../models/{}/crf.pkl'.format(proteasesOne), 'rb') as f:  #read byte
                        crf = pickle.load(f)
                    # y_pred = crf.predict(X_test_feats)
                    y_pred = crf.predict_marginals(X_test_feats)
                except:
                    print(f'The model of {proteasesOne} failed to load, please select another type of protease!')
                    sys.exit(1)
                model_type='CRF'
                # pre_scores =[int(item[0]) for sublist in y_pred for item in sublist]
                pre_scores =[round(float(item['1']),3) for sublist in y_pred for item in sublist]
                checkpoint_crf=1
            resultDF = pd.DataFrame() # 
            if model_type =='CRF':
                pdbid_pos=new_ids
            else:
                pdbid_pos=list(test_data.keys())
            
            pdbidL=[v.split('&')[0] for v in pdbid_pos]
            posL=[v.split('&')[1] for v in pdbid_pos]
            # print(model_type)
            if model_type !='CRF':
                pre_s=list(pre_scores[:,1])
                pre=[round(float(s),3) for s in pre_s]
                resultDF['prediction']=pre
            else:
                resultDF['prediction']=pre_scores
            resultDF.insert(0, 'Protease', [proteasesOne]*len(posL))
            resultDF.insert(1, 'IDs', pdbidL)
            resultDF.insert(2, 'Position', posL)
            # print(resultDF.head(10))
            if checkpoint_pre == 0:
                AllresultDF = resultDF
                checkpoint_pre = 1
            else:
                AllresultDF = AllresultDF.append(resultDF)
        AllresultDF.to_csv(args.outputpath + '/results.csv', index=False) 
    elif args.mode=="TrainYourModel": 
        if args.pdb_path==None:
            print('\nPlease provide the path of pdb files!\n')
            sys.exit(1)
        model_dir = args.outputpath+'/model_files/{}/'.format(args.protease)
        os.makedirs(model_dir, exist_ok=True)
        results_dir=args.outputpath
        protease=args.protease

        # Define loss function and optimizer
        loss_function = nn.CrossEntropyLoss()
        # Training models
        acc_dict={}
        mcc_dict={}
        if args.test_file != None:
            train_fasta_file_,train_data=read_inputfiles('train',args.inputpath,args,args.inputType,dataType='train') 
            test_fasta_file_,test_data=read_inputfiles('train',args.test_file,args,args.inputType,dataType='test') 
            train_indices,valid_indices=split_dataset(train_data,test_size=0.1)
            train_fasta_file_t,valid_fasta_file,data_train,data_valid=get_train_valid(train_fasta_file_,train_data,train_indices,valid_indices,args)
        else:
            input_fasta_file_,inputdata_=read_inputfiles('train',args.inputpath,args,args.inputType,dataType='train') 
            train_indices,test_indices=split_dataset(inputdata_,test_size=0.2)
            train_fasta_file_,test_fasta_file_,train_data,test_data=get_train_valid(input_fasta_file_,inputdata_,train_indices,test_indices,args)
            train_indices_,valid_indices=split_dataset(train_data,test_size=0.1)
            train_fasta_file_t,valid_fasta_file,data_train,data_valid=get_train_valid(train_fasta_file_,train_data,train_indices_,valid_indices,args)
        # training GNN model
        for graphType in ['knn','sequential','rball','ensemble']: #
            if graphType =='ensemble':
                try:
                    model_E =GVPEnsemble(1044, 128, 5, 0.95, 0.1).to(args.device)
                    print('*****Training {}-based model*****\n'.format(graphType))
                    print("The model contains {} parameters".format(sum(p.numel() for p in model_E.parameters() if p.requires_grad)))
                    optimizer = optim.Adam(model_E.parameters(), lr=args.learning_rate)
                    ensemble_based_model_acc,ensemble_based_model_mcc=train(args.n_epochs,model_E,optimizer,loss_function,train_dataloader_sequential,valid_dataloader_sequential,model_dir,results_dir,graphType,train_dataloader1=train_dataloader_knn,test_dataloader1=valid_dataloader_knn,train_dataloader2=train_dataloader_rball,test_dataloader2=valid_dataloader_rball)
                    print("Ensemble-based model's" f' valid acc:{round(ensemble_based_model_acc,3)}, valid mcc:{round(ensemble_based_model_mcc,3)}')
                    acc_dict['ensemble']=ensemble_based_model_acc
                    mcc_dict['ensemble']=ensemble_based_model_mcc
                except:
                    print('Failed to train ensemble model!')
                    continue
            else:
                model =GLM(1044, 128, 5, 0.95, 0.1).to(args.device)
                print('*****Training {}-based model*****\n'.format(graphType))
                print("The model contains {} parameters\n".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
                # load model
                # checkpoint = torch.load('model_5_0.65.pt')
                # model.load_state_dict(checkpoint,strict=False)
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
                
                if graphType =='sequential':
                    try:
                        valid_dataset_sequential  = ProteinGraphDataset(data_valid, range(len(data_valid)), valid_fasta_file,args,graph_type=graphType)
                        train_dataset_sequential  = ProteinGraphDataset(data_train, range(len(data_train)), train_fasta_file_t,args,graph_type=graphType)
                        valid_dataloader_sequential = DataLoader(valid_dataset_sequential , batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
                        train_dataloader_sequential = DataLoader(train_dataset_sequential , batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
                        sequential_based_model_acc,sequential_based_model_mcc=train(args.n_epochs,model,optimizer,loss_function,train_dataloader_sequential,valid_dataloader_sequential,model_dir,results_dir,graphType)
                        print("Sequential-based model's" f' valid acc:{round(sequential_based_model_acc,3)}, valid mcc:{round(sequential_based_model_mcc,3)}\n')
                        acc_dict['sequential']=sequential_based_model_acc
                        mcc_dict['sequential']=sequential_based_model_mcc
                    except:
                        print('Failed to train sequential-based model!')
                        continue
                elif graphType =='knn':
                    try:
                        valid_dataset_knn = ProteinGraphDataset(data_valid, range(len(data_valid)), valid_fasta_file,args,graph_type=graphType)
                        train_dataset_knn = ProteinGraphDataset(data_train, range(len(data_train)), train_fasta_file_t,args,graph_type=graphType)
                        valid_dataloader_knn = DataLoader(valid_dataset_knn, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
                        train_dataloader_knn = DataLoader(train_dataset_knn, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
                        knn_based_model_acc,knn_based_model_mcc=train(args.n_epochs,model,optimizer,loss_function,train_dataloader_knn,valid_dataloader_knn,model_dir,results_dir,graphType)
                        print("Knn-based model's" f' valid acc:{round(knn_based_model_acc,3)}, valid mcc:{round(knn_based_model_mcc,3)}\n')
                        acc_dict['knn']=knn_based_model_acc
                        mcc_dict['knn']=knn_based_model_mcc
                    except:
                        print('Failed to train knn-based model!')
                        continue
                elif graphType =='rball':
                    try:
                        valid_dataset_rball = ProteinGraphDataset(data_valid, range(len(data_valid)), valid_fasta_file,args,graph_type=graphType)
                        train_dataset_rball = ProteinGraphDataset(data_train, range(len(data_train)), train_fasta_file_t,args,graph_type=graphType)
                        valid_dataloader_rball = DataLoader(valid_dataset_rball, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
                        train_dataloader_rball = DataLoader(train_dataset_rball, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
                        rball_based_model_acc,rball_based_model_mcc=train(args.n_epochs,model,optimizer,loss_function,train_dataloader_rball,valid_dataloader_rball,model_dir,results_dir,graphType)
                        print("Rball-based model's" f' valid acc:{round(rball_based_model_acc,3)}, valid mcc:{round(rball_based_model_mcc,3)}\n')
                        acc_dict['rball']=rball_based_model_acc
                        mcc_dict['rball']=rball_based_model_mcc
                    except:
                        print('Failed to train rball-based model!')
                        continue

        # training pretrained-based model
        try:
            model_PBMLP =PBMLP(1044, 1024, 0.1,args.batch_size).to(args.device)#
            print('*****Training PBMLP-based model*****\n')
            print("The model contains {} parameters".format(sum(p.numel() for p in model_PBMLP.parameters() if p.requires_grad)))
            optimizer = optim.Adam(model_PBMLP.parameters(), lr=args.learning_rate)
            valid_dataset_PBMLP = ProteinSequenceDataset(data_valid, range(len(data_valid)), valid_fasta_file,args,graph_type='PBMLP')
            train_dataset_PBMLP = ProteinSequenceDataset(data_train, range(len(data_train)), train_fasta_file_t,args,graph_type='PBMLP')
            valid_dataloader_PBMLP = DataLoader(valid_dataset_PBMLP, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
            train_dataloader_PBMLP = DataLoader(train_dataset_PBMLP, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
            PBMLP_based_model_acc,PBMLP_based_model_mcc=train(args.n_epochs,model_PBMLP,optimizer,loss_function,train_dataloader_PBMLP,valid_dataloader_PBMLP,model_dir,results_dir,'PBMLP')
            print("PBMLP model's" f' valid acc:{round(PBMLP_based_model_acc,3)}, valid mcc:{round(PBMLP_based_model_mcc,3)}\n')
            acc_dict['PBMLP']=PBMLP_based_model_acc
            mcc_dict['PBMLP']=PBMLP_based_model_mcc  
        except:
            print('Failed to train PBMLP model!')
        # training CRF 
        if args.test_file != None:
            test_dataset_crf,test_y_crf,testdata_ids=structure_features(test_data,range(len(test_data)),test_fasta_file_,args.chain,args.dataset_path,args.pdb_path, protease,args,False)
            train_dataset_crf,train_y_crf,traindata_ids=structure_features(train_data,range(len(train_data)),train_fasta_file_,args.chain,args.dataset_path,args.pdb_path, protease,args,False)
            # testdata_ids=list(test_data.keys())
        else:
            data_x,data_y,data_ids=structure_features(inputdata_,range(len(inputdata_)),input_fasta_file_,args.chain,args.dataset_path,args.pdb_path, protease,args,False)
            train_indices,test_indices=split_dataset(data_ids)
            train_dataset_crf, test_dataset_crf = data_x[train_indices], data_x[test_indices]
            train_y_crf, test_y_crf = data_y[train_indices], data_y[test_indices]
            test_ids=[data_ids[i] for i in test_indices]      
        try:
            lowess = sm.nonparametric.lowess
            test_dataset_crf=MyLOWESS(test_dataset_crf,lowess)
            train_dataset_crf=MyLOWESS(train_dataset_crf,lowess)
            best_crf,crf_mcc,crf_acc=k_fold_cross_validation(train_dataset_crf,train_y_crf)
            acc_dict['CRF']=crf_acc
            mcc_dict['CRF']=crf_mcc 
            print("CRF-based model's" f' valid acc:{round(crf_acc,3)}, valid mcc:{round(crf_mcc,3)}\n')
            with open(model_dir+'crf.pkl', 'wb') as f:  
                pickle.dump(best_crf, f)
        except:
            print('Failed to train CRF model!')
        # To select a model according to mcc or acc of each trained graph-based model       
        dict_items=list(zip(mcc_dict.items(),acc_dict.items()))
        max_Gtype=sorted(dict_items, key = lambda kv:(kv[0][1], kv[1][1]),reverse = True)[0][0][0]
        # loading best model and test
        if max_Gtype =='ensemble':
            test_dataset_sequential  = ProteinGraphDataset(test_data, range(len(test_data)),  test_fasta_file_,args,graph_type='sequential')
            test_dataloader_sequential = DataLoader(test_dataset_sequential , batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
            test_dataset_knn = ProteinGraphDataset(test_data, range(len(test_data)),  test_fasta_file_,args,graph_type='knn')
            test_dataloader_knn = DataLoader(test_dataset_knn, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
            test_dataset_rball = ProteinGraphDataset(test_data, range(len(test_data)),  test_fasta_file_,args,graph_type='rball')
            test_dataloader_rball = DataLoader(test_dataset_rball, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
            test(max_Gtype,model_dir,results_dir,loss_function,test_dataloader_sequential,test_dataloader1=test_dataloader_knn,test_dataloader2=test_dataloader_rball)
            shutil.copy(args.outputpath+'/model_files/'+protease+'/model_ensemble.pth',args.outputpath+'/model_ensemble.pth')
        elif max_Gtype =='sequential':
            test_dataset_sequential  = ProteinGraphDataset(test_data, range(len(test_data)),  test_fasta_file_,args,graph_type='sequential')
            test_dataloader_sequential = DataLoader(test_dataset_sequential , batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
            test(max_Gtype,model_dir,results_dir,loss_function,test_dataloader_sequential)
            shutil.copy(args.outputpath+'/model_files/'+protease+'/model_sequential.pth',args.outputpath+'/model_sequential.pth')
        elif max_Gtype =='knn':
            test_dataset_knn = ProteinGraphDataset(test_data, range(len(test_data)),  test_fasta_file_,args,graph_type='knn')
            test_dataloader_knn = DataLoader(test_dataset_knn, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
            test(max_Gtype,model_dir,results_dir,loss_function,test_dataloader_knn)
            shutil.copy(args.outputpath+'/model_files/'+protease+'/model_knn.pth',args.outputpath+'/model_knn.pth')
        elif max_Gtype =='rball':
            test_dataset_rball = ProteinGraphDataset(test_data, range(len(test_data)),  test_fasta_file_,args,graph_type='rball')
            test_dataloader_rball = DataLoader(test_dataset_rball, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
            test(max_Gtype,model_dir,results_dir,loss_function,test_dataloader_rball)
            shutil.copy(args.outputpath+'/model_files/'+protease+'/model_rball.pth',args.outputpath+'/model_rball.pth')
        elif max_Gtype =='PBMLP':
            test_dataset_PBMLP = ProteinSequenceDataset(test_data, range(len(test_data)),  test_fasta_file_,args,graph_type='PBMLP')
            test_dataloader_PBMLP = DataLoader(test_dataset_PBMLP, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
            test(max_Gtype,model_dir,results_dir,loss_function,test_dataloader_PBMLP) 
            shutil.copy(args.outputpath+'/model_files/'+protease+'/model_PBMLP.pth',args.outputpath+'/model_PBMLP.pth')
        elif max_Gtype =='CRF':
            X_test_feats = [featuresProcess(sentence) for sentence in test_dataset_crf]
            y_pred = best_crf.predict(X_test_feats)
            y_pred_flat = [int(item[0]) for sublist in y_pred for item in sublist]
            acc=accuracy_score(test_y_crf, y_pred_flat)
            mcc=matthews_corrcoef(test_y_crf, y_pred_flat)
            se,sp=calculate_se_sp(list(test_y_crf),list(y_pred_flat))
            Recall=recall_score(list(test_y_crf),list(y_pred_flat))
            F1=f1_score(list(test_y_crf),list(y_pred_flat))
            Precision=precision_score(list(test_y_crf),list(y_pred_flat))
            resultDF = pd.DataFrame()
            resultDF['Model']=[max_Gtype]
            resultDF['Protease']=[protease]
            resultDF['ACC']=[round(acc,3)]
            resultDF['MCC']=[round(mcc,3)]
            resultDF['Recall']=[round(Recall,3)]
            resultDF['F1']=[round(F1,3)]
            resultDF['Precision']=[round(Precision,3)]
            resultDF['Sensitivity']=[round(se,3)]
            resultDF['Specificity']=[round(sp,3)]
            resultDF.to_csv(args.outputpath + '/performance.csv', index=False)  
            shutil.copy(args.outputpath+'/model_files/'+protease+'/crf.pkl',args.outputpath+'/crf.pkl')  
        if args.pre_file:
            tmp_fasta_file,pre_data=read_inputfiles('prediction',args.pre_file,args,args.inputType,crf=True,chain=args.chain)
            if max_Gtype =='ensemble':
                pre_dataset_sequential  = ProteinGraphDataset2(pre_data, range(len(pre_data)),  tmp_fasta_file,args,graph_type='sequential')
                pre_dataloader_sequential = DataLoader(pre_dataset_sequential , batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
                pre_dataset_knn = ProteinGraphDataset2(pre_data, range(len(pre_data)),  tmp_fasta_file,args,graph_type='knn')
                pre_dataloader_knn = DataLoader(pre_dataset_knn, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
                pre_dataset_rball = ProteinGraphDataset2(pre_data, range(len(pre_data)),  tmp_fasta_file,args,graph_type='rball')
                pre_dataloader_rball = DataLoader(pre_dataset_rball, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
                model = torch.load(args.outputpath+'/model_files/{}/model_ensemble.pth'.format(protease))
                pre_scores=Predict(model,pre_dataloader_sequential,graphType='ensemble',test_loader1=pre_dataloader_knn,test_loader2=pre_dataloader_rball)
            elif max_Gtype =='sequential':
                pre_dataset_sequential  = ProteinGraphDataset2(pre_data, range(len(pre_data)),  tmp_fasta_file,args,graph_type='sequential')
                pre_dataloader_sequential = DataLoader(pre_dataset_sequential , batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
                model = torch.load(args.outputpath+'/model_files/{}/model_sequential.pth'.format(protease))
                pre_scores=Predict(model,pre_dataloader_sequential,graphType='sequential')
            elif max_Gtype =='knn':
                pre_dataset_knn = ProteinGraphDataset2(pre_data, range(len(pre_data)),  tmp_fasta_file,args,graph_type='knn')
                pre_dataloader_knn = DataLoader(pre_dataset_knn, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
                model = torch.load(args.outputpath+'/model_files/{}/model_knn.pth'.format(protease))
                pre_scores=Predict(model,pre_dataloader_knn,graphType='knn')
            elif max_Gtype =='rball':
                pre_dataset_rball = ProteinGraphDataset2(pre_data, range(len(pre_data)),  tmp_fasta_file,args,graph_type='rball')
                pre_dataloader_rball = DataLoader(pre_dataset_rball, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
                model = torch.load(args.outputpath+'/model_files/{}/model_rball.pth'.format(protease))
                pre_scores=Predict(model,pre_dataloader_rball,graphType='rball')
            elif max_Gtype =='PBMLP':
                pre_dataset_PBMLP = ProteinSequenceDataset1(pre_data, range(len(pre_data)),  tmp_fasta_file,args,graph_type='PBMLP')
                pre_dataloader_PBMLP = DataLoader(pre_dataset_PBMLP, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
                model = torch.load(args.outputpath+'/model_files/{}/model_PBMLP.pth'.format(protease))
                pre_scores=Predict(model,pre_dataloader_PBMLP,graphType='PBMLP')
            elif max_Gtype =='CRF':
                pdbpath=os.path.dirname(args.pre_file)
                pre_dataset,pre_y,_=structure_features(pre_data,range(len(pre_data)),tmp_fasta_file,args.chain,args.dataset_path,pdbpath, protease,args,True)
                pre_dataset==MyLOWESS(pre_dataset,lowess)
                X_pre_feats=[featuresProcess(sentence) for sentence in pre_dataset]
                y_pred = best_crf.predict(X_pre_feats)
                pre_scores =[int(item[0]) for sublist in y_pred for item in sublist]

            pdbid_pos=list(pre_data.keys())
            resultDF1 = pd.DataFrame()
            pdbidL=[v.split('&')[0] for v in pdbid_pos]
            posL=[v.split('&')[1] for v in pdbid_pos]

            if max_Gtype !='CRF':
                pre=list(pre_scores[:,1])
            else:
                pre=pre_scores
            resultDF1['Prediction']=pre 
            resultDF1.insert(0, 'Protease', [protease]*len(posL))
            resultDF1.insert(1, 'Model', [max_Gtype]*len(posL))
            resultDF1.insert(2, 'IDs', pdbidL)
            resultDF1.insert(3, 'Position', posL)
            resultDF1.to_csv(args.outputpath + '/results.csv', index=False)
        
        os.system('rm -r {}/train_set.fasta'.format(args.outputpath))
        os.system('rm -r {}/valid_set.fasta'.format(args.outputpath))
        os.system('rm -r {}/test.fasta'.format(args.outputpath))
        os.system('rm -r {}/train.fasta'.format(args.outputpath))
        os.system(f'rm -r {args.outputpath}/model_files')
    elif args.mode=='UseYourOwnModel':
        print(args.model_file)

        if args.pre_file=='None':
            print('Please provide the model file!')
            sys.exit(1)
        if args.protease=='None':
            print('Please provide the protease type!')
            sys.exit(1)

        protease=args.protease 
        
        if 'ensemble.pth' in args.model_file:
            model_type='ensemble'
            tmp_fasta_file,pre_data=read_inputfiles('prediction',args.pre_file,args,args.inputType,crf=None,chain=args.chain)
            pre_dataset_sequential  = ProteinGraphDataset2(pre_data, range(len(pre_data)),  tmp_fasta_file,args,graph_type='sequential')
            pre_dataloader_sequential = DataLoader(pre_dataset_sequential , batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
            pre_dataset_knn = ProteinGraphDataset2(pre_data, range(len(pre_data)),  tmp_fasta_file,args,graph_type='knn')
            pre_dataloader_knn = DataLoader(pre_dataset_knn, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
            pre_dataset_rball = ProteinGraphDataset2(pre_data, range(len(pre_data)),  tmp_fasta_file,args,graph_type='rball')
            pre_dataloader_rball = DataLoader(pre_dataset_rball, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
            
            try:
                model = torch.load(args.model_file)
                pre_scores=Predict(model,pre_dataloader_sequential,graphType='ensemble',test_loader1=pre_dataloader_knn,test_loader2=pre_dataloader_rball)
            except:
                print('Please provide the correct model!')
                sys.exit(1)
            
        elif 'sequential.pth' in args.model_file:
            model_type='sequential'
            tmp_fasta_file,pre_data=read_inputfiles('prediction',args.pre_file,args,args.inputType,crf=None,chain=args.chain)
            pre_dataset_sequential  = ProteinGraphDataset2(pre_data, range(len(pre_data)),  tmp_fasta_file,args,graph_type='sequential')
            pre_dataloader_sequential = DataLoader(pre_dataset_sequential , batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
            try:
                model = torch.load(args.model_file)
                pre_scores=Predict(model,pre_dataloader_sequential,graphType='sequential')
            except:
                print('Please provide the correct model!')
                sys.exit(1)
        elif 'knn.pth' in args.model_file:
            model_type='knn'
            tmp_fasta_file,pre_data=read_inputfiles('prediction',args.pre_file,args,args.inputType,crf=None,chain=args.chain)
            pre_dataset_knn = ProteinGraphDataset2(pre_data, range(len(pre_data)),  tmp_fasta_file,args,graph_type='knn')
            pre_dataloader_knn = DataLoader(pre_dataset_knn, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
            try:
                model = torch.load(args.model_file)
                pre_scores=Predict(model,pre_dataloader_knn,graphType='knn')
            except:
                print('Please provide the correct model!')
                sys.exit(1)
            
        elif 'rball.pth' in args.model_file:
            model_type='rball'
            tmp_fasta_file,pre_data=read_inputfiles('prediction',args.pre_file,args,args.inputType,crf=None,chain=args.chain)
            pre_dataset_rball = ProteinGraphDataset2(pre_data, range(len(pre_data)),  tmp_fasta_file,args,graph_type='rball')
            pre_dataloader_rball = DataLoader(pre_dataset_rball, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
            try:
                model = torch.load(args.model_file)
                pre_scores=Predict(model,pre_dataloader_rball,graphType='rball')
            except:
                print('Please provide the correct model!')
                sys.exit(1) 
        elif 'PBMLP.pth' in args.model_file:
            model_type='PBMLP'
            tmp_fasta_file,pre_data=read_inputfiles('prediction',args.pre_file,args,args.inputType,crf=None,chain=args.chain)
            pre_dataset_PBMLP = ProteinSequenceDataset1(pre_data, range(len(pre_data)),  tmp_fasta_file,args,graph_type='PBMLP')
            pre_dataloader_PBMLP = DataLoader(pre_dataset_PBMLP, batch_size = args.batch_size, shuffle=False, drop_last=False, prefetch_factor=2)
            try:
                model = torch.load(args.model_file)
                pre_scores=Predict(model,pre_dataloader_PBMLP,graphType='PBMLP')
            except:
                print('Please provide the correct model!')
                sys.exit(1)
            
        elif 'crf.pkl' in args.model_file:
            model_type='CRF'
            tmp_fasta_file,pre_data=read_inputfiles('prediction',args.pre_file,args,args.inputType,crf=True,chain=args.chain)
            lowess = sm.nonparametric.lowess
            pdbpath=os.path.dirname(args.pre_file)
            pre_dataset,pre_y,_=structure_features(pre_data,range(len(pre_data)),tmp_fasta_file,args.chain,args.dataset_path,pdbpath, protease,args,True)
            pre_dataset==MyLOWESS(pre_dataset,lowess)
            X_pre_feats=[featuresProcess(sentence) for sentence in pre_dataset]
            with open(args.model_file, 'rb') as f: 
                best_crf = pickle.load(f)
            y_pred = best_crf.predict(X_pre_feats)
            pre_scores =[int(item[0]) for sublist in y_pred for item in sublist]
        else:
            print('Please provide the original model file(the output of the TrainYourModel module)!')
            sys.exit(1)
        pdbid_pos=list(pre_data.keys())
        resultDF1 = pd.DataFrame()
        pdbidL=[v.split('&')[0] for v in pdbid_pos]
        posL=[v.split('&')[1] for v in pdbid_pos]
        if model_type !='CRF':
            pre=list(pre_scores[:,1])
            pre_s=[round(v,3) for v in pre]
            resultDF1['Prediction']=pre_s 
        else:
            pre=pre_scores
            resultDF1['Prediction']=pre 
        resultDF1.insert(0, 'Protease', [protease]*len(posL))
        resultDF1.insert(1, 'Model', [model_type]*len(posL))
        resultDF1.insert(2, 'IDs', pdbidL)
        resultDF1.insert(3, 'Position', posL)
        resultDF1.to_csv(args.outputpath + '/results.csv', index=False)
    print('Complete!')
    
    os.system('rm -r {}/proteases_structure_data'.format(args.outputpath))
    os.system('rm -r {}/proteases_prottrans'.format(args.outputpath))
    if os.path.exists(args.outputpath + '/naccess' ):   
        os.system('rm -r {}/naccess'.format(args.outputpath))
    if os.path.exists(args.outputpath + '/temp_crf.fasta' ):
        os.system('rm -r {}/temp_crf.fasta'.format(args.outputpath))
    if os.path.exists(args.outputpath + '/temp.fasta' ):
        os.system('rm -r {}/temp.fasta'.format(args.outputpath))
    
    
    
    
