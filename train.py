import os
import sys
import shutil
from collections import defaultdict
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from util import CrossEntropyLabelSmooth

def train_model(model, dl, lr, epochs, classify_dim=17, best_top1_acc=0, save_path = "", feature_num=10000):
    #####set optimizer and criterin#####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) ##
    criterion = nn.MSELoss().to(device)
    criterion_smooth_cty = CrossEntropyLabelSmooth().to(device)
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        for batch_sample in zip(*(dl)):
            optimizer.zero_grad()
            for i in range(len(dl)):
                data_label = batch_sample[i]
                data = data_label['data']
                data = Variable(data)
                data = torch.reshape(data,(data.size(0),-1)).to(device)
                label = data_label['label']
                label = Variable(label).to(device)

                data_reconstruction, data_prediction = model(data)

                cty_loss = criterion_smooth_cty(data_prediction, label)
                ae_loss = criterion(data,data_reconstruction)
                ae_loss.backward(retain_graph=True)
                cty_loss.backward(retain_graph=True)

            optimizer.step()
    return model