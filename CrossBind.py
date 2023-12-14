# -*- coding: utf-8 -*-

import os
import sys

import pprint
import random
import argparse
import warnings
import importlib

import numpy as np
import torch.utils.data
import torch.optim as optim

from tqdm import tqdm
from datetime import datetime

from utils.lr_scheduler import get_scheduler
from utils.logger import setup_logger
from cfgs.config import cfg, cfg_from_yaml_file, backup_files
from utils.metrics import evaluators
from collections import Counter
from models.sparseconvunet_inference import base_model, Mix_net,  atom_attention 

import esm
from utils.eval_util import AverageMeter,print_metrics,get_mertics,list_to_tuple
warnings.filterwarnings('ignore')
pp = pprint.PrettyPrinter()


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='/CrossBind/cfgs/SparseConv-Cath-Decoys-Clf-Only.yaml', help='specify the config for training')
    parser.add_argument('--log_dir', type=str, default=None, help='specify the save direction for training')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--num_workers', type=int, default=None, help='number of data loading workers')
    parser.add_argument('--gpu', type=int, nargs='+', default=(0,), help='specify gpu devices')
    parser.add_argument('--debug', action='store_true', default=False, help='whether not to down sample PC')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    cfg.cfg_file = args.cfg_file
    cfg.debug = args.debug
    cfg.gpu = args.gpu
    cfg.log_dir = args.log_dir

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    random.seed(cfg.manualSeed)
    torch.manual_seed(cfg.manualSeed)
    np.random.seed(cfg.manualSeed)
    torch.cuda.manual_seed(cfg.manualSeed)

    if cfg.log_dir:
        cfg.log_dir = os.path.join('logs', f'{cfg.dataset}', f'{cfg.log_dir}')
    else:
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        cfg.log_dir = os.path.join('logs',  f'{cfg.dataset}', '%s-%s'%(cfg.model, now))

    return cfg


def load_checkpoint(config, model, optimizer, scheduler):
    checkpoint = torch.load(os.path.join(config.log_dir, 'checkpoint', 'current.pth'), map_location='cpu')
    config.from_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("=> loaded successfully '{}/checkpoint/current.pth' (epoch {})".format(config.log_dir, checkpoint['epoch']+1))

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, optimizer, scheduler):
    logger.info('==> Saving...')
    os.makedirs(os.path.join(config.log_dir, 'checkpoint'), exist_ok=True)

    state = {
        'config': config,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(config.log_dir, 'checkpoint', 'current.pth'))
    logger.info("Saved in {}".format(os.path.join(config.log_dir, 'checkpoint', f'ckpt_epoch_{epoch}.pth')))


def main(config):
    logger.info(config)

    if config.dataset == 'cath_decoys':
        from datasets.cath_decoys.protein_dataset import get_dataset
    else:
        raise NotImplementedError

    # dataloader
    logger.info('Load Dataset...')
    dataset = get_dataset(config, logger)
    # dataset.pssm_hmm_dssp_feature()
    dataset.init_pipeline()
    dataset.init_trainloader()
    dataset.init_valloader()
    dataset.pssm_hmm_dssp_feature()

    train_data_loader = dataset.train_data_loader
    val_data_loader = dataset.val_data_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    logger.info('Load Model...')
    f_model = importlib.import_module('models.' + 'sparseconvunet_inference')
    model = f_model.get_model(config).to(device)
    
    esm_model = base_model().to(device)
    mix_model = Mix_net().to(device)
    attention_model = atom_attention().to(device)

    ALPHABET = esm_model.get_alphabet()
    batch_converter = ALPHABET.get_batch_converter()

    criterion = f_model.get_loss(config).to(device)



    if torch.cuda.device_count() > 1:
        print("Number of GPUs:", torch.cuda.device_count())
    ids=[]
    for i in range(torch.cuda.device_count()):
        ids.append(i)
    print(ids)
    # model = torch.nn.DataParallel(model,device_ids=ids).to(device)


    logger.info("#model parameters {}".format(sum(x.numel() for x in model.parameters())))



    optimizer = optim.Adam([
        {'params': model.parameters(),'lr': 1e-5},
        {'params': esm_model.parameters(),'lr': 1e-5},
        {'params': mix_model.parameters(),'lr': 1e-4},
        {'params': attention_model.parameters(),'lr': 1e-4},
    ])


    scheduler = get_scheduler(optimizer, config)

    # load model
    try:
        load_checkpoint(config, model, optimizer, scheduler)
    except:
        logger.info("Training model from scratch...")
# 
    model.load_state_dict(torch.load('/nvme/xusheng1/Linglin/resource/ProteinDecoy-main/models/DNA_127_Structure.pkl'))
    # model.load_state_dict(torch.load('/nvme/xusheng1/Linglin/resource/ProteinDecoy-main/models/Test_181.pkl'))


    for epoch in range(config.from_epoch, config.nepoch):

        logger.info('')
        logger.info('======>>>>> Online epoch: #%d, lr=%f <<<<<======' % (epoch+1, scheduler.get_lr()[0]))
        esm_model.train()
        mix_model.train()
        model.train()
        attention_model.train()
        # self_attention_model.train()
        evaluator = evaluators(config)
        predict_auc = []
        label_auc = []
        with tqdm(total=len(train_data_loader)) as pbar:
            for data in train_data_loader:
                # torch.cuda.synchronize()
                # tess = torch((data['reg_labels']))
                for k in data.keys():
                    if k in ['features', 'cls_labels', 'reg_labels']:
                        data[k] = data[k].to(device)


                ### MSA feature
                all_feature = dataset.all_feature[(data['file_name'])[0]]
                all_feature = all_feature.astype(np.float32)
                all_feature = torch.from_numpy(all_feature)
                
                ### ESM2
                batch_data = [((data['file_name'])[0],(data['protein_seq'])[0])]
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
                batch_tokens = batch_tokens[:,1:-1]


            
                esm_out = esm_model(batch_tokens.to(device))
                output,batch_label = model(data,device,attention_model)
                Final_output = mix_model(output.to(device), esm_out.to(device),all_feature.to(device),animo_feature,data,device)

                end_points = data
                end_points['residual'] =  Final_output
                end_points['batch_label'] = batch_label


                end_points = criterion(end_points,device)
                optimizer.zero_grad()
                
                loss = end_points['loss']
                desc_str,predict_all,label_all = evaluator.add_batch(end_points,'train')

                predict_auc=np.append(predict_auc,np.array(predict_all.detach().cpu().numpy()))
                label_auc=np.append(label_auc,np.array(label_all.detach().cpu().numpy()))

                loss.backward()

                optimizer.step()
                


                pbar.set_description(desc_str)
                pbar.update(1)
                
        scheduler.step(epoch)
        # torch.cuda.synchronize()
        print('total_num' , len(train_data_loader))
        evaluator.print_batch_metric(logger, epoch, len(train_data_loader),predict_auc,label_auc )
        del evaluator


        evaluator = evaluators(config)
        model.eval()
        esm_model.eval()
        mix_model.eval()
        attention_model.eval()

        predict_auc = []
        label_auc = []

        for data in val_data_loader:
            # torch.cuda.synchronize()
            for k in data.keys():
                if k in ['features', 'cls_labels', 'reg_labels']:
                    data[k] = data[k].to(device)


            all_feature = dataset.all_feature[(data['file_name'])[0]]
            all_feature = all_feature.astype(np.float32)
            all_feature = torch.from_numpy(all_feature)


            batch_data = [((data['file_name'])[0],(data['protein_seq'])[0])]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)

            batch_tokens = batch_tokens[:,1:-1]
            esm_out = esm_model(batch_tokens.to(device))
            output,batch_label = model(data,device,attention_model)
            # output,batch_label = model(data,device,self_attention_model)
            Final_output = mix_model(output.to(device), esm_out.to(device),all_feature.to(device),animo_feature,data,device)

            end_points = data
            end_points['residual'] = Final_output
            end_points['batch_label'] = batch_label

            end_points = criterion(end_points,device)
            evaluator.add_batch(end_points,'test')
            del end_points
            predict_auc=np.append(predict_auc,np.array(predict_all.detach().cpu().numpy()))
            label_auc=np.append(label_auc,np.array(label_all.detach().cpu().numpy()))

 
        evaluator.print_batch_metric(logger, epoch, len(val_data_loader),predict_auc,label_auc , 'test')

        del evaluator

    # logger.info("End.")

if __name__ == "__main__":
    config = parse_config()
    torch.set_num_threads(4)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    os.makedirs(config.log_dir, exist_ok=True)
    os.environ["JOB_LOG_DIR"] = config.log_dir
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 
    logger = setup_logger(output=config.log_dir, name='%s' % (config.model))

    main(config)
