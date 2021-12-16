#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn

def pairwise_loss(outputs1, outputs2, label1, label2, class_num=1.0):
    sim = (label1.float().mm(label2.float().t()) > 0.5).float()

    dot_xy = outputs1.mm(outputs2.t())

    positive_ind = sim.data >= 0.5
    negative_ind = sim.data < 0.5
    #class_num = negative_ind.float().sum() / positive_ind.float().sum()

    log_loss = torch.log(1 + torch.exp(dot_xy)) - sim * dot_xy
 
    loss = (class_num * positive_ind.float() * log_loss + negative_ind.float() * log_loss).sum()

    return loss / (( positive_ind.float()).float().sum() + negative_ind.float().sum())