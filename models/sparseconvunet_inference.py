import torch
import torch.nn as nn

import sparseconvnet as scn
from collections import Counter
import numpy as np
import torch.nn.functional as F  
import esm
from torch.nn import init
from sklearn import preprocessing
def get_model(config):
    return SparseConvUnet(config)

def get_loss(config):
    return Loss(config)

fii = [-0.44985162,  0.73240308,  0.45873502, -0.26693213,  1.2191339 ,
0.31799025, -0.70646635,  0.06760465,  0.80724017, -0.41348552,
-1.09356942,  0.44781314,  0.91290933,  0.18138941,  0.15670218,
0.17694924,  0.30954355,  1.31420132,  0.60408098, -0.54580296,
1.3 ]

aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']



class Mix_net(nn.Module):
    def __init__(self):
        super().__init__()


        self.lstm = nn.LSTM(
            input_size=4,# 
            hidden_size=12,# 
            batch_first=True,# 
        )

        self.fc1 = nn.Linear(54, 1)
        self.fc1_esm = nn.Linear(640, 128)
        self.fc1_mix = nn.Linear(1088, 128)
        self.fc1_fu = nn.Linear(14*32, 128)
        self.fc1_mix_hhm = nn.Linear(448, 128) ##502, 950 , 448 , 1782

        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,1)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.w1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
    def forward(self, aa_features,esm_out,all_feature,data_dict,device):
        
        

        ###########
        ######### fusion
        w1 = torch.sigmoid(self.w1)
        esm_out = esm_out.squeeze(0)
        fusion = torch.add((1-w1)*esm_out , w1*aa_features)

        final_input = torch.concat((fusion,all_feature),1)
        # final_input = fusion

   
        out = F.elu(self.fc1_mix_hhm(final_input))

        out = F.elu(self.fc2(out))  
        # out = self.dropout(out)

        output = torch.sigmoid(self.fc3(out))   

        return output

class base_model(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self):
        super().__init__()    # 

        self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        # self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        # self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        # self.esm_model.eval()

        self.fc1 = torch.nn.Linear(1280,448)#.to('cuda:0')
        self.dropout = torch.nn.Dropout(p=0.5)


    def get_alphabet(self):
        return self.esm_alphabet

        
    def forward(self,batch_tokens):

        # with torch.no_grad():
        results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        out = token_representations
        out = self.fc1(token_representations)
        
        return out

class Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, data_dict,device):
        loss_fun = torch.nn.BCELoss()
        loss = 0
        aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']


        reg_pred = data_dict['residual']
        reg_labels = data_dict['reg_labels']
        reg_labels = reg_labels.unsqueeze(1)

 
        label_tensor = data_dict['batch_label'].unsqueeze(1)


        ######### AA Level 
        # reg_loss =  loss_fun(reg_pred_new.float(),label_tensor.float().to(device))#.cuda())
        reg_loss =  loss_fun(reg_pred.float(),label_tensor.float().to(device))#.cuda())

        loss = reg_loss
        data_dict['reg_loss'] = reg_loss
        data_dict['label'] = reg_labels
        data_dict['binary_label'] = label_tensor

        data_dict['loss'] = loss

    return data_dict

class atom_attention(nn.Module):
    def __init__(self):
        super().__init__()


        self.attention = nn.Linear(448 , 14,bias=False)
        self.embedding = nn.Linear(448, 448)

    def forward(self,x):
        fee = torch.flatten(x,0,1)
        atom_atten = self.attention(fee)

        atten_par =  torch.sigmoid(atom_atten)

        after_attention = atten_par.unsqueeze(1) * x 
        fee2 = torch.flatten(after_attention,0,1)
        output = self.embedding(fee2)
        output = output.reshape(14,32)
        # return output.unsqueeze(1)
        return output


class SparseConvUnet(nn.Module):
    def __init__(self, config):
        super().__init__()


        self.config = config
        m = config.m
        input_dim = 30 if config.use_coords else 27
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(3, config.full_scale, mode=config.mode)).add(
           scn.SubmanifoldConvolution(3, input_dim, m, 3, False)).add(
               scn.UNet(dimension=3,
                        reps=config.block_reps,
                        nPlanes=[m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m],
                        residual_blocks=config.block_residual,
                        )).add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(3))



        self.lstm = nn.LSTM(
            input_size=1088,# 
            hidden_size=1280,# 
            batch_first=True,# 
        )

        if 'regression' in config:
            self.fc1 = nn.Linear(14*m, 1)
            # self.fc1 = nn.Linear(1*m, 1)
            self.fc2 = nn.Linear(128,64)
            self.fc3 = nn.Linear(64,1)
            self.dropout = torch.nn.Dropout(p=0.5)

            # self.attention = nn.Linear(14*m , 14)
            # self.embedding = nn.Linear(14*m,14*m)

            self.project = nn.Sequential(
                nn.Linear(32, 32, bias = False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 32, bias = False)
                ) 

    def forward(self, data_dict,device,attention_model):

        # with torch.no_grad():
        am = data_dict['coords'][:,0:3]
        input_batch = [
            data_dict['coords'],
            data_dict['features']
        ]

        feature = self.sparseModel(input_batch)

            

        if 'regression' in self.config:
            
            atom_num = data_dict['atom_num']
            label_binary = data_dict['label_binary']
            index = 0

            # print('model^^^^^^^^^^^^')
            idn = 0
            final_feature = []
            final_feature_attention = []
            final_label = []

        
            for z in (atom_num):
                idn += 1
                aa_feature = torch.zeros(len(z) ,14 ,feature.shape[1])
                aa_feature_attention = torch.zeros(len(z) ,14 ,feature.shape[1])
                aa_feature_mean = torch.zeros(len(z)  ,feature.shape[1])            
                j = 0
                
                for i in z:
                 
                    aa_feature[j,:i,:] = feature[index:(index+i),:]
                    atom_wise = aa_feature[j,:].clone()
                    index = index + i 
                    aa_feature_attention = aa_feature
                    j += 1

                final_feature_attention.append(aa_feature_attention)
                final_feature.append(aa_feature)
                # final_feature.append(aa_feature_mean)
            feature_batch = torch.cat(final_feature , 0)
            aa_feature = torch.flatten(feature_batch,1,2)

            feature_batch_attention = torch.cat(final_feature_attention , 0)
            aa_feature_attention = torch.flatten(feature_batch_attention,1,2)

            binary_label = data_dict['label_binary'] 
            for z in binary_label:
                final_label.append(z)
            batch_label_1= np.concatenate(final_label , 0)
            tet_cuda = torch.from_numpy(batch_label_1).to(device)

        return aa_feature_attention,tet_cuda




