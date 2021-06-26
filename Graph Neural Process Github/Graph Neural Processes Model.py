# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:23:30 2020

@author: Professor Junbin Gao
"""
# Thanks to zfjsail for providing open-source Variational Graph Autoencoders Pytorch implementation.
# This code is based on the above code, link: https://github.com/zfjsail/gae-pytorch.git

from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch


from model import GNP_Encoder, GNP_Decoder, InnerProductDecoder
from optimizer import loss_function3
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score, ct_split


#%%

torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--hiddenEnc', type=int, default=32, help='Number of units in hidden layer of Encoder.')
parser.add_argument('--z_dim', type=int, default=32, help='Dimension of latent code Z.')
parser.add_argument('--hiddenDec', type=int, default=64, help='Number of units in hidden layer of Decoder.')
parser.add_argument('--outDimDec', type=int, default=32, help='Output Dimension of the Decoder.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset_str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--n_z_samples', type=int, default=10, help='Number of Z samples')

#%%

args = parser.parse_args()

torch.manual_seed(args.seed)

#%%
def sample_z(mu, std, n):
    """Reparameterisation trick."""
    eps = torch.autograd.Variable(std.data.new(n,args.z_dim).normal_())
    return mu + std * eps 

def KLD_gaussian(mu_q, std_q, mu_p, std_p):
    """Analytical KLD between 2 Gaussians."""
    qs2 = std_q**2 + 1e-16
    ps2 = std_p**2 + 1e-16
    
    return (qs2/ps2 + ((mu_q-mu_p)**2)/ps2 + torch.log(ps2/qs2) - 1.0).sum()*0.5
    
def gae_for(args):
    adj, features = load_data(args.dataset_str)
    print("Using {} dataset".format(args.dataset_str))
    n_nodes, feat_dim, = features.shape
    
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    
    adj = adj_train
    adj_deep_copy = adj
    print(adj.shape)
    
    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    
    encoder = GNP_Encoder(feat_dim, args.hiddenEnc, args.z_dim, args.dropout)
    decoder = GNP_Decoder(feat_dim + args.z_dim, args.hiddenDec, args.outDimDec, args.dropout)
    innerDecoder = InnerProductDecoder(args.dropout, act=lambda x: x)
    optimizer = torch.optim.Adam(list(decoder.parameters())+list(encoder.parameters()), args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=1)
    
    train_loss = []
    val_ap = []
    for epoch in range(args.epochs):             
        t = time.time()
        encoder.train()
        decoder.train()
        innerDecoder.train()        
        
        np.random.seed(args.seed)
        adj_context  = ct_split(adj_deep_copy )
        adj_context_norm = preprocess_graph(adj_context)        
        
        c_z_mu, c_z_logvar = encoder(features, adj_context_norm)
        ct_z_mu, ct_z_logvar = encoder(features, adj_norm)
        
        #Sample a batch of zs using reparam trick.
        zs = sample_z(ct_z_mu, torch.exp(ct_z_logvar), args.n_z_samples)
        
        # Get the predictive distribution of y*
        mu, std = decoder(features, zs)

        emb = torch.mean(mu, dim = 1)
        pred_adj = innerDecoder(emb)
        
        #Compute loss and backprop
        loss = loss_function3(preds=pred_adj, labels=adj_label, norm=norm, pos_weight=torch.tensor(pos_weight))  + KLD_gaussian(ct_z_mu, torch.exp(ct_z_logvar), c_z_mu, torch.exp(c_z_logvar))
        
        optimizer.zero_grad()
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()        
        scheduler.step()
        
        hidden_emb = emb.data.numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
        
        
        train_loss.append(cur_loss)
        val_ap.append(ap_curr)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t))
     
    print("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    return roc_score, ap_score

if __name__ == '__main__':
        gae_for(args)
