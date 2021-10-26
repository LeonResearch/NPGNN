import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from layers import GraphConvolution

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class Discriminator(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Discriminator, self).__init__()
        self.dc_den1 = torch.nn.Linear(hidden_dim2, hidden_dim3)
        self.dc_den1.bias.data.fill_(0.0)
        self.dc_den1.weight.data = torch.normal(0.0, 0.001, [hidden_dim3, hidden_dim2])
        self.dc_den2 = torch.nn.Linear(hidden_dim3, hidden_dim1)
        self.dc_den2.bias.data.fill_(0.0)
        self.dc_den2.weight.data = torch.normal(0.0, 0.001, [hidden_dim1, hidden_dim3])
        self.dc_output = torch.nn.Linear(hidden_dim1, 1)
        self.dc_output.bias.data.fill_(0.0)
        self.dc_output.weight.data = torch.normal(0.0, 0.001, [1,hidden_dim1])
        self.act = F.relu
    def forward(self, inputs):
        dc_den1 = self.act(self.dc_den1(inputs))
        dc_den2 = self.act(self.dc_den2(dc_den1))
        output = self.dc_output(dc_den2)
        return output

# Graph Neural Process (GNP) Encoder
class GNP_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, z_dim, dropout):
        super(GNP_Encoder, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, z_dim, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, z_dim, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

        
    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)


    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        z_logvar = torch.log(torch.mean(std))
        z_mu = torch.mean(mu)
        return z_mu, z_logvar
            
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z_mu, z_logvar = self.reparameterize(mu, logvar)
        return z_mu, z_logvar


# Graph Neural Process (GNP) Decoder
class GNP_Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, init_func = None):
        super(GNP_Decoder, self).__init__()
        
        # Question: Why only self.l1_size = hidden_dim, but no self.something = input_dim
        self.l1_size = hidden_dim
        
        # Two layer of fully connected NNs
        self.l1 = torch.nn.Linear(input_dim, self.l1_size)
        self.l2 = torch.nn.Linear(self.l1_size, output_dim)
        
        if init_func is not None:
            init_func(self.l1.weight)
            init_func(self.l2.weight)
        
        # Question: Here why sigmoid as activation Function But not Relu? Cause it is classification?
        self.a = torch.nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        #Glorot initialization
        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.xavier_uniform_(self.l2.weight)
    
    def forward(self, x_pred, z):
        """x_pred: No. of data points, by x_dim
        z: No. of samples, by z_dim
        """
        zs_reshaped = z.unsqueeze(-1).expand(z.shape[0], z.shape[1], x_pred.shape[0]).transpose(1,2)
        xpred_reshaped = x_pred.unsqueeze(0).expand(z.shape[0], x_pred.shape[0], x_pred.shape[1])
        
        xz = torch.cat([xpred_reshaped, zs_reshaped], dim=2)
        # Question: Dimension = 2 ?!
        
        # Question: Why return 0.005?
        return self.l2(self.a(self.l1(xz))).squeeze(-1).transpose(0,1), 0.005
