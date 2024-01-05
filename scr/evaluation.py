import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix,classification_report,matthews_corrcoef,precision_score,f1_score,recall_score,roc_auc_score
import pickle,math
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils import calculate_se_sp,y_trueTOnumpy,preClass

def evaluate_family_classifier(model, test_loader,loss_function=None,graphType='knn', batch_size=None,
                               mode='test', device='cpu',final=False,dir=None,test_loader1=None,test_loader2=None): 

    model.eval()

    with torch.no_grad():
        losses = []
        trainresults=[]
        y_true_list2 = []
        results = {}
        preds_mapper=[{'0':0,'1':1}]
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
                target=data.y
                out1 = model(data,data1,data2)
            elif graphType=='PBMLP':
                seq_pro,seq_ind,target = data
                seq_pro,seq_ind,target=seq_pro.to(device),seq_ind.to(device),target.to(device)
                target=target.squeeze()
                out1 = model(seq_pro,seq_ind) 
            else:
                data = data.to(device)
                target=data.y
                out1 = model(data)
            loss = loss_function(out1[0], target.long())
            
            losses.append(loss.item())

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
            
            y_true_list2 += list(target)

            y_pred = torch.nn.Softmax(dim=1)(out1[0]).cpu().detach().numpy()
            trainresults.append(y_pred)
        scores= np.vstack(trainresults)
        results = [np.hstack(task_res) for task_res in results.values()]
        results = [
            np.vectorize(preds_mapper[task_idx].get)(task_res.astype(str))
            for task_idx, task_res in enumerate(results)
        ]
        
        avg_loss = np.mean(losses)
  
        y_true_tensor2 = torch.stack(y_true_list2)
        y_true=y_true_tensor2.cpu().numpy()
        yT=y_trueTOnumpy(y_true)
        y_prdclass=preClass(results)
        accs=[accuracy_score(yT[:,i],y_prdclass[:,i]) for i in range(1)]
        mccs=[matthews_corrcoef(yT[:,i],y_prdclass[:,i]) for i in range(1)]
        if final:
            y_true_tensor2 = torch.stack(y_true_list2)
            y_true=y_true_tensor2.cpu().numpy()
            yT=y_trueTOnumpy(y_true)
            y_prdclass=preClass(results)
            # print('Loss:',round(avg_loss,6))
            accs=[accuracy_score(yT[:,i],y_prdclass[:,i]) for i in range(1)]
            mccs=[matthews_corrcoef(yT[:,i],y_prdclass[:,i]) for i in range(1)]
            se,sp=calculate_se_sp(list(yT[:,0]),list(y_prdclass[:,0]))
            One_hot_encoder=OneHotEncoder(sparse=False)
            y_test_onehot_label=One_hot_encoder.fit_transform(yT[:,0].reshape(len(list(yT[:,0])),1))
            auc=roc_auc_score(y_test_onehot_label,scores,average='weighted')
            
            Recall=recall_score(list(yT[:,0]),list(y_prdclass[:,0]))
            F1=f1_score(list(yT[:,0]),list(y_prdclass[:,0]))
            Precision=precision_score(list(yT[:,0]),list(y_prdclass[:,0]))

            return round(accs[0],3),round(mccs[0],3),round(Recall,3),round(F1,3),round(Precision,3),round(se,3),round(sp,3),round(auc,3)
        else:
            return avg_loss,accs[0],mccs[0]


