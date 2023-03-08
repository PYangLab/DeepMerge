import os
import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable

from model import DeepMerge,DeepMerge_3modality
from train import train_model
from util import setup_seed, MyDataset,ToTensor, read_h5_data, read_fs_label, get_encodings, get_decodings, compute_zscore, compute_log2
import argparse


parser = argparse.ArgumentParser("DeepMerge")
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--dataset', type=str, default="dataset", help='seed')
parser.add_argument('--modality1', type=str, default="NULL", help='modality name')
parser.add_argument('--modality2', type=str, default="NULL", help='modality name')
parser.add_argument('--modality3', type=str, default="NULL", help='modality name')
############# for data build ##############
parser.add_argument('--modality1_path', metavar='DIR', default='NULL', help='path to modality1 data')
parser.add_argument('--modality2_path', metavar='DIR', default='NULL', help='path to modality2 data')
parser.add_argument('--modality3_path', metavar='DIR', default='NULL', help='path to modality3 data')
parser.add_argument('--cty_path', metavar='DIR', default='NULL', help='path to cell type ')
parser.add_argument('--batch_path', metavar='DIR', default='NULL', help='path to batch information')

##############  for training #################
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=1, help='num of training epochs')
parser.add_argument('--lr', type=float, default=0.02, help='init learning rate')

############# for model build ##############
parser.add_argument('--z_dim', type=int, default=100, help='the number of neurons in latent space')
parser.add_argument('--hidden_modality1', type=int, default=185, help='the number of neurons for modality1 layer')
parser.add_argument('--hidden_modality2', type=int, default=30, help='the number of neurons for modality2 layer')
parser.add_argument('--hidden_modality3', type=int, default=185, help='the number of neurons for modality3 layer')

args = parser.parse_args()



dataset = args.dataset
seed = args.seed
setup_seed(seed) ### set random seed in order to reproduce the result
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# parameters for training #################
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
# parameters for model build ##############
z_dim = args.z_dim
hidden_modality1 = args.hidden_modality1
hidden_modality2 = args.hidden_modality2
modality1_path = args.modality1_path
modality2_path = args.modality2_path
modality1 = args.modality1
modality2 = args.modality2

if args.modality3_path !='NULL':
    hidden_modality3 = args.hidden_modality3
    modality3_path = args.modality3_path
    modality3 = args.modality3
cty_path = args.cty_path
batch_path = args.batch_path

# read_data
modality1_all = read_h5_data(modality1_path)
modality2_all = read_h5_data(modality2_path)
if args.modality3_path !='NULL':
    modality3_all = read_h5_data(modality3_path)
cty_all = read_fs_label(cty_path)
batch_all = read_fs_label(batch_path)

batch_dim = max(batch_all)+1
classify_dim = max(cty_all)+1
nfeatures_modality1 = modality1_all.size(1)
nfeatures_modality2 = modality2_all.size(1)
if args.modality3_path !='NULL':
    nfeatures_modality3 = modality3_all.size(1)

dl_train = list()
dl_test = list()
for i in range(batch_dim):
    modality1_temp = compute_zscore(compute_log2(modality1_all[batch_all==i,:]))
    modality2_temp = compute_zscore(compute_zscore(compute_log2(modality2_all[batch_all==i,:])))
    if args.modality3_path !='NULL':
        modality3_temp = compute_zscore(compute_zscore(compute_log2(modality3_all[batch_all==i,:])))
    cty_temp = cty_all[batch_all==i]
    
    if args.modality3_path =='NULL':
        data_temp = torch.cat((modality1_temp,modality2_temp),1)
    else:
        data_temp = torch.cat((modality1_temp,modality2_temp,modality3_temp),1)
        
    transformed_dataset_temp = MyDataset(data_temp, cty_temp)
    dl_train_temp = DataLoader(transformed_dataset_temp, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    dl_test_temp = DataLoader(transformed_dataset_temp, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    dl_train.append(dl_train_temp)
    dl_test.append(dl_test_temp)
    
if args.modality3_path =='NULL':
    model = DeepMerge(nfeatures_modality1,nfeatures_modality2, hidden_modality1, hidden_modality2, z_dim, classify_dim).to(device)
else:
    model = DeepMerge_3modality(nfeatures_modality1, nfeatures_modality2, nfeatures_modality3, hidden_modality1, hidden_modality2, hidden_modality3, z_dim, classify_dim).to(device)
    
# train the network
model = train_model(model, dl_train, lr=lr, epochs=epochs, classify_dim = classify_dim)

# obtain the reconstructed data
embedding = list()
if args.modality3_path =='NULL':
    reconstruction_modality1 = list()
    reconstruction_modality2 = list()
else:
    reconstruction_modality3 = list()
for i in range(batch_dim):
    embedding_temp = get_encodings(model, dl_test[i])
    reconstruction_temp = get_decodings(model, dl_test[i])
    if args.modality3_path =='NULL':
        reconstruction_modality1_temp = reconstruction_temp[:,0:nfeatures_modality1]
        reconstruction_modality2_temp = reconstruction_temp[:,nfeatures_modality1:]
    else:
        reconstruction_modality1_temp = reconstruction_temp[:,0:nfeatures_modality1]
        reconstruction_modality2_temp = reconstruction_temp[:,nfeatures_modality1:(nfeatures_modality1+nfeatures_modality2)]
        reconstruction_modality3_temp = reconstruction_temp[:,(nfeatures_modality1+nfeatures_modality2):]
            
    embedding.append(embedding_temp)
    if args.modality3_path =='NULL':
        reconstruction_modality1.append(reconstruction_modality1_temp)
        reconstruction_modality2.append(reconstruction_modality2_temp)
    else:
        reconstruction_modality3.append(reconstruction_modality3_temp)
    
####################save#################
modality_list = range(0, modality1_all.size(0))
cell_name = ['cell_{}'.format(b) for b in modality_list]  
pd.DataFrame(torch.cat(embedding,0).cpu().numpy(), index = cell_name).to_csv("./result/{}/data_merged.csv".format(dataset))
pd.DataFrame(torch.cat(reconstruction_modality1,0).cpu().numpy(), index = cell_name).to_csv("./result/{}/{}_merged.csv".format(dataset,modality1))
pd.DataFrame(torch.cat(reconstruction_modality2,0).cpu().numpy(), index = cell_name).to_csv("./result/{}/{}_merged.csv".format(dataset,modality2))
if args.modality3 !='NULL':
    pd.DataFrame(torch.cat(reconstruction_modality3,0).cpu().numpy(), index = cell_name).to_csv("./result/{}/{}_merged.csv".format(dataset,modality3))
