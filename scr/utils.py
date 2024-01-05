import pandas as pd
from Bio import SeqIO
import numpy as np
from sklearn.metrics import matthews_corrcoef,accuracy_score 
# from sklearn.model_selection import train_test_split
# import statsmodels.api as sm
from sklearn_crfsuite import CRF
from sklearn.model_selection import KFold
def preClass(predict_classes):
    predictClassedDF=pd.DataFrame()
    predictClassedDF['predLabel_0']=predict_classes[0]
    predictClassedDF['predLabel_0']=predictClassedDF['predLabel_0'].apply(int)
    return predictClassedDF.values
def y_trueTOnumpy(y_t):
    y_validDf=pd.DataFrame(y_t,columns=['label_0'])
    y_validDf['label_0']=y_validDf['label_0'].apply(int)
    return y_validDf.values

def split_dataset(data, test_size=0.2, random_state=42):
    # 将样本id按照pdbid分组
    sample_ids=list(data.keys())
    Y=np.array([int(data[k][1]) for k in list(data.keys())])
    grouped_samples = {}
    for i, sample_id in enumerate(sample_ids):
        # pdbid = sample_id.split("_")[0]
        pdbid=sample_id.split('&')[0].split('_')[-1]
        if pdbid not in grouped_samples:
            grouped_samples[pdbid] = []
        grouped_samples[pdbid].append(i)

    # 初始化训练集和测试集的索引
    train_indices = []
    test_indices = []

    # 针对每个pdbid的样本分配到训练集或测试集
    for pdbid, indices in grouped_samples.items():
        if len(indices) > 1:
            test_ratio=(len(test_indices)/len(sample_ids))
            # print(test_ratio)
            if test_ratio<=test_size:
                test_indices.extend(indices)
            else:
                train_indices.extend(indices)

    # 根据已划分好的训练集和测试集的比例，将只有一个样本的pdbid加入到训练集或测试集
    for pdbid, indices in grouped_samples.items():
        if len(indices) == 1:
            # 如果该pdbid只有一个样本，根据当前训练集和测试集的正负标签比例决定加入到训练集或测试集
            single_sample_index = indices[0]
            single_sample_label = Y[single_sample_index]
            
            pos_count_train = np.sum(Y[train_indices] == 1)
            neg_count_train = len(train_indices) - pos_count_train

            pos_count_test = np.sum(Y[test_indices] == 1)
            neg_count_test = len(test_indices) - pos_count_test

            if single_sample_label == 1:
                if (pos_count_test / (pos_count_train)) < test_size:
                    test_indices.extend(indices)
                else:
                    train_indices.extend(indices)
            else:
                if (neg_count_test / (neg_count_train)) < test_size:
                    test_indices.extend(indices)
                else:
                    train_indices.extend(indices)

    # 根据索引获取划分后的数据集
    # X_train, X_test = X[train_indices], X[test_indices]
    # Y_train, Y_test = Y[train_indices], Y[test_indices]
    # test_ids=[sample_ids[i] for i in test_indices]
    return train_indices,test_indices#X_train, X_test, Y_train, Y_test,test_ids

def get_train_valid(in_file,in_data,train_indices,valid_indices,args):
    ids_=list(in_data.keys())
    # 将样本id按照pdbid分组
    grouped_samples = {}
    for i, sample_id in enumerate(ids_):
        # pdbid = sample_id.split("_")[0]
        # pdbid=sample_id.split('&')[0].split('_')[-1]
        if sample_id not in grouped_samples:
            grouped_samples[sample_id] = []
        grouped_samples[sample_id].append(i)
    # print("train_indices",train_indices,type(train_indices[0]))
    train_ids=[]
    valid_ids=[]
    for id_, indices in grouped_samples.items(): 
        # print(indices)
        for ind in indices:
            if ind in train_indices:
                train_ids.append(id_.strip())
            elif ind in valid_indices:
                valid_ids.append(id_.strip())
    # print("train_ids",train_ids)
    train_file_=args.outputpath+'/train_set.fasta'
    valid_file_=args.outputpath+'/valid_set.fasta'
    f=open(train_file_,'w')
    f1=open(valid_file_,'w')
    data_train={}
    data_valid={}
    for record in SeqIO.parse(in_file,'fasta'):
        des=str(record.description)
        name=des.split()[0].strip()
        # print("name",name)
        seq=str(record.seq)
        label=int(des.split()[1])
        pos=int(name.split('&')[1])
        # print(name,pos,label)
        if name in train_ids:
            f.write('>'+des+'\n')
            f.write(seq+'\n')
            data_train[name] = (pos,label,seq)
        elif name in valid_ids:
            f1.write('>'+des+'\n')
            f1.write(seq+'\n')
            data_valid[name] = (pos,label,seq)
        else:
            print('get_train_valid:error')
    f.close()
    f1.close()
    return train_file_,valid_file_,data_train,data_valid

def featuresProcess(sentence):
    return  [{'x_'+str(i):x for i,x in enumerate(sentence)}] 
def MyLOWESS(dataset,lowess):
    # print(dataset.shape)
    x_values=[i for i in range(dataset.shape[1])]
    data=[]
    for i in range(dataset.shape[0]):
        data.append(lowess(list(dataset[i]),x_values, frac=0.05, it=3,return_sorted=False))
    return np.vstack(data)
def calculate_se_sp(labels, scores, cutoff=0.5, po_label=1):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(scores)):
        if labels[i] == po_label:
            # if scores[i] >= cutoff:
            if scores[i] ==1:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            # if scores[i] < cutoff:
            if scores[i] ==0:
                tn = tn + 1
            else:
                fp = fp + 1

    Sensitivity= tp / (tp + fn) if (tp + fn) != 0 else 0
    Specificity= tn / (fp + tn) if (fp + tn) != 0 else 0
    return Sensitivity,Specificity

def train_crf(X_train, y_train):
    # 定义CRF模型

    crf = CRF(
        algorithm='lbfgs',
        c1=1.0,
        c2=1e-3,
        max_iterations=50,
        all_possible_transitions=True
    )

    crf.fit(X_train, y_train)

    return crf

def k_fold_cross_validation(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    models = []
    mccs = []
    accs=[]
    for train_index, test_index in kf.split(X):
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = np.array(y)[train_index], np.array(y)[test_index]

        # 提取训练数据的特征和标签
        X_train_feats = [featuresProcess(sentence) for sentence in X_train]
        y_train_labels = [str(label) for label in y_train]
        
        X_valid_feats = [featuresProcess(sentence) for sentence in X_valid]
        # y_valid_labels = [str(label) for label in y_test]

        # 训练CRF模型
        crf = train_crf(X_train_feats, y_train_labels)
        models.append(crf)

        # 在验证集上评估性能
        y_pred = crf.predict(X_valid_feats)
        y_pred_flat = [int(item[0]) for sublist in y_pred for item in sublist]
        
        mcc=matthews_corrcoef(list(y_valid), y_pred_flat)
        # f1_score = flat_f1_score(y_valid_labels, y_pred, average='weighted')
        mccs.append(mcc)
        acc=accuracy_score(list(y_valid), y_pred_flat)
        accs.append(acc)

    best_model_index = np.argmax(mccs)
    best_model = models[best_model_index]

    return best_model,mccs[best_model_index],accs[best_model_index]