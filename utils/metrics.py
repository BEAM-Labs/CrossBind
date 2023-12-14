# -*- coding: utf-8 -*-
'''
@Author: Xu Yan
@File: metrics.py
@Time: 2021/6/29 21:07
'''

from sklearn.metrics import roc_auc_score,matthews_corrcoef
from sklearn.metrics import auc, precision_recall_curve,roc_curve
from sklearn import preprocessing

import numpy as np
from collections import Counter
from utils.eval_util import AverageMeter,print_metrics,get_mertics,list_to_tuple
import math

fii = [-0.44985162,  0.73240308,  0.45873502, -0.26693213,  1.2191339 ,
        0.31799025, -0.70646635,  0.06760465,  0.80724017, -0.41348552,
       -1.09356942,  0.44781314,  0.91290933,  0.18138941,  0.15670218,
        0.17694924,  0.30954355,  1.31420132,  0.60408098, -0.54580296,
        1.1 ]
# aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']

# fii = [0,0,0,0,0,0,-1,0,0,0,0,-1,-1,1,1,0,0,0,0,0,0]
aa_list = ['G', 'A', 'V', 'L', 'I', 'M', 'C', 'S', 'T', 'N', 'Q', 'D', 'E', 'K', 'R', 'H', 'F', 'Y', 'W', 'P', 'X']

scalera = preprocessing.MinMaxScaler(feature_range = (0.8,1.2))
fii_input = np.array(fii).reshape(-1,1)
new_fii = scalera.fit_transform(fii_input)

class evaluators():
    def __init__(self, config):
        self.config = config
        self.total_loss = 0
        self.cls_loss = 0
        self.reg_loss = 0
        self.bin_true = 0
        self.bin_total = 0
        self.output_l = []
        self.label_l = []
        self.auc = 0
        self.auc_all = 0
        self.aucr  = 0

        self.predict_auc = []
        self.predict_auc_new = []
        self.label_auc = []

        self.predict_true = []
        self.label_true = []

        self.my_confusion_matrix = {
        'TP': 0.0001,
        'TN': 0.0001,
        'FP': 0.0001,
        'FN': 0.0001
    }

        self.my_confusion_matrix_new = {
        'TP': 0.0001,
        'TN': 0.0001,
        'FP': 0.0001,
        'FN': 0.0001
    }
        # self.predict_all = 0
        # self.label_all = 0


    def down_sample(self,data,seq):
        # data = [ 1,2,3,4,5,6,7,8,9,10]
        idx = [-2,-1,0,1,2]
        z = 0
        A_name = []
        new_output = data.copy()
        AA_name = seq
        # for i in seq:
        #     A_name.append((i))
        # AA_name = np.array(A_name)

        for i in range ((data.shape[0])):
            new = []
            location = []

            for j in idx:
                if i < 2:
                    j = 0
                elif i > len(data)-3:
                    j = 0
                if data[j+i] > 0.5:
                    location.append(i+j)
                    new.append(1)
                    z += 1
                else:
                    new.append(0)
                    location.append(i+j)
                    z += 1
                    
            count_x = dict(Counter(new))

            for k, v in count_x.items():
                if k == 1:
                    if v == 4:
                        id = new.index(0)
                        aa_loca = location[id]
                        a = aa_list.index(AA_name[aa_loca])
                    
                        # if fii[a] == 1:
                        if  fii[a] > 0.7:
                            new_output[i] = new_output[i] * 1.2 

                    elif v == 1:
                        id = new.index(1)
                        aa_loca = location[id]
                        a = aa_list.index(AA_name[aa_loca])

                        # if fii[a] == -1:
                        if fii[a] < -0.5:
                            new_output[i] = new_output[i] * 0.8

                
        return new_output

    def add_batch(self, out_dict,mode):
        disc_str = 'Loss: %.4f' % out_dict['loss']
        self.total_loss += out_dict['loss'].item()

        if 'cls_loss' in out_dict:
            self.cls_loss += out_dict['cls_loss'].item()
            disc_str += ' Cls_Loss: %.4f' % out_dict['cls_loss']

            bin_pred = out_dict['bin'].argmax(1).cpu().numpy()
            bin_gt = out_dict['cls_labels'].cpu().numpy()
            bin_true = len(bin_pred[bin_pred == bin_gt])
            bin_total = len(bin_pred)
            disc_str += ' Cls_ACC: %.2f' % (bin_true/bin_total)

            self.bin_true += bin_true
            self.bin_total += bin_total

            output = out_dict['bin']
            label = out_dict['label']
            # print(label)
            # output = np.array(output.cpu())
            self.output_l=np.append(self.output_l,output.detach().numpy())
            self.label_l=np.append(self.label_l,label.detach().numpy())

            # print(self.label_l.shape,self.output_l.shape)
            # print(Counter(self.label_l))
            # print(Counter(label))
            self.predict_all = (output.detach().numpy())
            self.label_all = (label.detach().numpy())
            self.auc = roc_auc_score(label.detach().numpy(),output.detach().numpy())
            # print(self.auc)
            disc_str += ' auc: %.4f' % (self.auc)
            self.auc_all += self.auc

        if 'reg_loss' in out_dict:
            self.reg_loss += out_dict['reg_loss'].item()
            disc_str += ' Reg_Loss: %.4f' % out_dict['reg_loss']

            output = out_dict['residual']
            label = out_dict['label']
            label_binary = out_dict['binary_label']
            #### Atom
            # self.auc = roc_auc_score(label.detach().cpu().numpy(),output.detach().cpu().numpy())
            # precision_list, recall_list, thresholds = precision_recall_curve(
            # label.detach().cpu().numpy(), output.detach().cpu().numpy())
            # self.aucr_sample = auc(recall_list, precision_list)


            # self.predict_auc=np.append(self.predict_auc,np.array(output.detach().cpu().numpy()))
            # self.label_auc=np.append(self.label_auc,np.array(label.detach().cpu().numpy()))

            # pred_label = (output > 0.5)
            # true_label = (label)
            # true_negative = torch.sum(((pred_label == 0) & (true_label == 0)))
            # true_positive = torch.sum(((pred_label == 1) & (true_label == 1)))
            # false_negative = torch.sum(((pred_label == 0) & (true_label == 1)))
            # false_positive = torch.sum(((pred_label == 1) & (true_label == 0)))

            ##### AA 

            self.auc = roc_auc_score(label_binary.detach().cpu().numpy(),output.detach().cpu().numpy())
            precision_list, recall_list, thresholds = precision_recall_curve(
                        label_binary.detach().cpu().numpy(), output.detach().cpu().numpy())
            self.aucr_sample = auc(recall_list, precision_list)

            self.predict_auc=np.append(self.predict_auc,np.array(output.detach().cpu().numpy()))
            new_out = output.detach().cpu().numpy()
            A_name = []
            xx = out_dict['protein_seq']
            
            for i in xx[0]:
                A_name.append((i))
            AA_name = np.array(A_name)
            # new_output = np.zeros(output.shape)
            new_output = new_out
            aaaaaaa = new_fii
            # for i in range (output.shape[0]):
            #     a = aa_list.index(AA_name[i])
                # if fii[a] < -0.6:
                #     new_output[i] = new_out[i] * 0.5
                # if fii[a] > 1:
                #     new_output[i] = new_out[i] * 1.5     

                # new_output[i] = new_out[i] + (1/(1 + math.exp(-fii[a])))
                # new_output[i] = new_out[i] * new_fii[a]
                # idxs = np.where(new_output>1) 
                # idxs2 = np.where(new_output<0) 
                # new_output[idxs] = 1
                # new_output[idxs2] = 0
            
            # self.predict_auc_new=np.append(self.predict_auc_new,np.array(new_output))

            # if mode == 'test':
            #     new_output = self.down_sample(output.detach().cpu().numpy(),AA_name)
            # new_out = output
            self.predict_auc_new=np.append(self.predict_auc_new,new_output)

            self.label_auc=np.append(self.label_auc,np.array(label_binary.detach().cpu().numpy()))
        #     import pandas as pd
        # # fpr = fpr.reshape(-1,1)
        #     pd.DataFrame(np.array(self.label_auc)).to_csv('/nvme/xusheng1/Linglin/resource/test.csv')
            
        #     ##
            pred_label = (output > 0.5)

            # binary_preds = [1 if score >= output else 0 for score in output]
            # pred_label = (output > 0.57)
            true_label = (label_binary)
            true_negative = torch.sum(((pred_label == 0) & (true_label == 0)))
            true_positive = torch.sum(((pred_label == 1) & (true_label == 1)))
            false_negative = torch.sum(((pred_label == 0) & (true_label == 1)))
            false_positive = torch.sum(((pred_label == 1) & (true_label == 0)))
            self.my_confusion_matrix['FN'] += false_negative.item()
            self.my_confusion_matrix['FP'] += false_positive.item()
            self.my_confusion_matrix['TN'] += true_negative.item()
            self.my_confusion_matrix['TP'] += true_positive.item()

            if mode == 'test':
                # if  out_dict['file_name'] == ['6ymw_B']:

                accuracy = (true_negative.item() + true_positive.item())/ (output.shape[0])
                # if accuracy > 98:
                #     print(out_dict['file_name'], accuracy)      
                #     accuracy = (true_negative.item() + true_positive.item())/ (output.shape[0])
                #     binary_preds = [1 if score >= 0.5 else 0 for score in output]
                #     cc = np.array(binary_preds)
                #     cc = cc.reshape(1,-1)
                #     # cc = cc.T
                #     np.savetxt('/nvme/xusheng1/Linglin/GraphSite-master/demo/esm.txt',cc,fmt='%d')

                #     lines = open('/nvme/xusheng1/Linglin/GraphSite-master/demo/esm.txt').readlines() #打开文件，读入每一行
        
                #     fp = open('/nvme/xusheng1/Linglin/GraphSite-master/demo/New_test2.txt','w') #打开你要写得文件pp2.txt
                #     for s in lines:
                #         for ss in s:
                #             if ss != ' ':
                #             # fp.write(ss.replace('    ','')) # replace是替换，write是写入
                #                 fp.write(ss)
                #     fp.close() # 关闭文件


                
                if accuracy > 0.98:
                    print(out_dict['file_name'], accuracy)

            true_label_new = true_label.detach().cpu()
            pred_label_new = (torch.from_numpy(new_output) > 0.65)
            # pred_label = (output > 0.65)
            # pred_label_new = true_label
            
            true_negative = torch.sum(((pred_label_new == 0) & (true_label_new == 0)))
            true_positive = torch.sum(((pred_label_new == 1) & (true_label_new == 1)))
            false_negative = torch.sum(((pred_label_new == 0) & (true_label_new == 1)))
            false_positive = torch.sum(((pred_label_new == 1) & (true_label_new == 0)))
            self.my_confusion_matrix_new['FN'] += false_negative.item()
            self.my_confusion_matrix_new['FP'] += false_positive.item()
            self.my_confusion_matrix_new['TN'] += true_negative.item()
            self.my_confusion_matrix_new['TP'] += true_positive.item()


            self.predict_true = np.append(self.predict_true , pred_label.cpu()*1)
            self.label_true = np.append(self.label_true , true_label.cpu())
            ####
            disc_str += ' auc: %.4f' % (self.auc)
            self.auc_all += self.auc
            self.aucr += self.aucr_sample
        return disc_str , output, label_binary

    def print_batch_metric(self, logging, epoch, total_num, predict_all_l , label_all_l,mode='Training'):
        logging.info('======>>>>> %s Metrics of epoch: #%d <<<<<======' % (mode, epoch + 1))
        logging.info('Total Loss: %.4f' % (self.total_loss/total_num))
        
        print_metrics(self.my_confusion_matrix)
        acc,precision,recall,f1,spe = get_mertics(self.my_confusion_matrix)
        print('SPe' , spe)
        print('pre,rec,F1' ,round(precision,4),round(recall,4),round(f1,4)  )
        
        AUC = self.auc_all/total_num
        AUCR = self.aucr/total_num
        MCC = matthews_corrcoef(self.label_true,self.predict_true)
        # print('mean_auc,aucr' , AUC,AUCR)
        # print('amino_auc,aucr' , AUC,AUCR)
        # print('atom_auc,aucr' , AUC,AUCR)
        # print('old_scool_amino,aucr' , AUC,AUCR)
        print('old' , AUC,AUCR)
        print('mcc' , MCC)
        auc_roc = roc_auc_score(self.label_auc, self.predict_auc)
        # import pandas as pd
        # pd.DataFrame(self.label_auc).to_csv('/nvme/xusheng1/Linglin/resource/label.csv')
        # pd.DataFrame(self.predict_auc).to_csv('/nvme/xusheng1/Linglin/resource/predict.csv')

        fpr , tpr, thread = roc_curve(self.label_auc, self.predict_auc)
        # np.savetxt('/nvme/xusheng1/Linglin/resource/fqr.txt',fpr,delimiter=',')
        # np.savetxt('/nvme/xusheng1/Linglin/resource/tpr.txt',tpr,delimiter=',')
        # fpr = fpr.reshape(-1,1)


        # import matplotlib.pyplot as plt
        # plt.figure()
        # lw = 2
        # plt.plot(fpr, tpr, color='darkorange',
        #         lw=lw)
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.0])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.show()
        # plt.savefig('/nvme/xusheng1/Linglin/resource/roc1.png')

        auc_roc_new = roc_auc_score(self.label_auc, self.predict_auc_new)
        precision_list, recall_list, thresholds = precision_recall_curve(
        self.label_auc, self.predict_auc)

        precision_list_new, recall_list_new, thresholds_new = precision_recall_curve(
        self.label_auc, self.predict_auc_new)


        auc_precision_recall = auc(recall_list, precision_list)
        auc_precision_recall_new = auc(recall_list_new, precision_list_new)

        f1_scores = (2 * precision_list * recall_list) / (precision_list + recall_list)
        best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
        best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

        f1_scores_new = (2 * precision_list_new * recall_list_new) / (precision_list_new + recall_list_new)
        best_f1_score_new = np.max(f1_scores_new[np.isfinite(f1_scores_new)])
        best_f1_score_index_new = np.argmax(f1_scores_new[np.isfinite(f1_scores_new)])

        print('new_f1' , best_f1_score_new , thresholds[best_f1_score_index_new])
        print('Best_F1' , best_f1_score , thresholds[best_f1_score_index])
        print('proroor',auc_roc_new , auc_precision_recall_new)
        print('new',auc_roc, auc_precision_recall )

        # if auc_roc > 95.5:
        #     pd.DataFrame(fpr).to_csv('/nvme/xusheng1/Linglin/resource/fqr.csv')
        #     pd.DataFrame(tpr).to_csv('/nvme/xusheng1/Linglin/resource/tpr.csv')
        
        if self.cls_loss > 0:
            logging.info('Total Classification Loss: %.4f' % (self.cls_loss/total_num))
            acc = self.bin_true/self.bin_total
            # logging.info('Total Classification Accuracy: %.4f' % acc)

        if self.reg_loss > 0:
            logging.info('Total Regression Loss: %.4f' % (self.reg_loss/total_num))
        return auc_roc

import torch
b, n = 2, 100
coords = torch.randint(106, [b, n, 3])

in_channels = 3
feats = torch.rand(b * n, in_channels)
a = 1