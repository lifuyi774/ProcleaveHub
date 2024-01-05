# ProcleaveHub: an AlphaFold2-aware geometric graph learning framwork for protease-specific substrate cleavage site prediction.
## Introduction
ProcleaveHub is available as a webserver and a stand-alone software package at http://procleavehub.unimelb-biotools.cloud.edu.au/. Sorry

## Environment
* Anaconda
* python 3.7.13

## Dependency

* pandas    1.3.5
* numpy		1.21.6
* scikit-learn    0.23.2
* sklearn-crfsuite    0.3.6
* statsmodels    0.13.2
* scipy    1.5.4
* biopython    1.79
* torch    1.12.1+cu116
* torch-cluster    1.6.0
* torch-scatter    2.0.9
* torch-sparse    0.6.15
* torch-geometric    2.1.0.post1
* torchvision    0.13.1+cull6
* torchaudio    0.12.1+cu116
* torch-spline-conv    1.2.1
* transformers    4.26.1

## ProtTrans
You need to prepare the pretrained language model ProtTrans to run ProcleaveHub:  
Download the pretrained ProtT5-XL-UniRef50 model ([guide](https://github.com/agemagician/ProtTrans)). 
### Note:
Please change the ProtTrans path in "get_prottrans" and "get_prottrans1" of "feature.py".

## naccess
You need to install naccess to run ProcleaveHub:
"unzip naccess.zip"
"csh install.scr"
### Note:
Please change the naccess path in "get_dssp_feature" of "feature.py".

## models
Model in the "prediction" module is available from http://procleavehub.unimelb-biotools.cloud.edu.au/.

## Usage

To get the information the user needs to enter for help, run:
    python ProcleaveHub.py --help
 or
    python ProcleaveHub.py -h

as follows:

>python ProcleaveHub.py -h

optional arguments:
  --mode MODE    Three modes can be used: prediction, TrainYourModel, UseYourOwnModel, only select one mode each time.
  --protease PROTEASE    The protease you want to predict cleavage to, eg:A01.001 Or if you want to build a new model, please create a name. There should no space in the model name.
  --chain CHAIN    A or B
  --outputpath OUTPUTPATH    The path of output.
  --inputpath INPUTPATH    The path of the training set file. Each entry in the training set contains the label, cleavage site position, pdb_id (uniprot accession), and chain type, which are separated by commas.
  --test_file TEST_FILE    he path of the test set file. Each entry in the training set contains the label, cleavage site position, pdb_id (uniprot accession), and chain type, which are separated by commas.
  --pre_file PRE_FILE    The path of the pdb file to predict.
  --model_file MODEL_FILE    The path of the model file to deploy.
  --pdb_path PDB_PATH    The path of the pdb files for the training (test or validation) set.
  --inputType INPUTTYPE    pdb
  --batch_size BATCH_SIZE    Batch size
  --device DEVICE    cpu or cuda
  --n_epochs N_EPOCHS    The number of training epoch.
  --learning_rate LEARNING_RATE    Learning rate.
  --early_stopping EARLY_STOPPING    Number of epochs for early stopping.

## Examples:

### Prediction:
```python ProcleaveHub.py --mode prediction --protease N10.002 --outputpath ./results --pre_file ../example/prediction.pdb --inputType pdb```
### TrainYourModel:
```python ProcleaveHub.py --mode TrainYourModel --protease N10.002 --inputpath ../example/train.txt --test_file ../example/test.txt --outputpath ./results --pre_file ../example/prediction.pdb --pdb_path ../example/pdbfiles```
### useYourModel:
```python ProcleaveHub.py --mode UseYourOwnModel --protease N10.002 --outputpath ./results --pre_file ../example/prediction.pdb --model_file ../example/model_rball.pth --inputType pdb```
## Output:
When the task is prediction or UseYourOwnModel, the result of the program is the test result of the model; while when the task is TrainYourModel, the results of the program include a file of best-model, a file of the best-model's performance, and a file of of the prediction results.
