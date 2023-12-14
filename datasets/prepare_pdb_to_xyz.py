# -*- coding: utf-8 -*-
'''
@Author: Xu Yan
@File: prepare_pdb_to_xyz.py
@Time: 2021/6/27 21:18
'''


import os
import sys
sys.path.append(os.path.abspath('/mnt/petrelfs/jinglinglin/Linglin/SpConv/ProteinDecoy-main/datasets/cath_decoys/'))
import numpy as np

from tqdm import tqdm

retval = os.getcwd()
os.chdir('/mnt/petrelfs/jinglinglin/Linglin/SpConv/ProteinDecoy-main/datasets/cath_decoys/')
a = os.listdir('pdb')
retval1 = os.getcwd()

print('Transfering P2B to XYZ...')
with tqdm(total=len(os.listdir('pdb'))) as pbar:
    for name in os.listdir('pdb'):
        file = name
        name = file[:-4]
        os.makedirs(os.path.join('xyz', name), exist_ok=True)
        print(name)

        # b = os.path.join('/mnt/petrelfs/jinglinglin/Linglin/SpConv/ProteinDecoy-main/datasets/cath_decoys/pdb', name)
        # for file in os.listdir(os.path.join('pdb', name)):
        # for file in os.listdir('pdb'):
        in_file = '/mnt/petrelfs/jinglinglin/Linglin/SpConv/ProteinDecoy-main/datasets/cath_decoys/pdb/%s' % ( file)
        out_file = '/mnt/petrelfs/jinglinglin/Linglin/SpConv/ProteinDecoy-main/datasets/cath_decoys/xyz/%s/%s' % (name, file.replace('pdb', 'xyz'))
        os.system('/mnt/petrelfs/jinglinglin/Linglin/SpConv/ProteinDecoy-main/LIG_Tool-master/util/PDB_To_XYZ -i %s -a 1 -o %s' % (in_file, out_file))
        pbar.set_description('%s/%s' % (name, file))
        pbar.update(1)

os.system('cp -r /mnt/petrelfs/jinglinglin/Linglin/SpConv/ProteinDecoy-main/datasets/cath_decoys/xyz/ /mnt/petrelfs/jinglinglin/Linglin/SpConv/ProteinDecoy-main/datasets/cath_decoys/xyz_label')

print('Mapping labels to each residue...')
with tqdm(total=len(os.listdir('xyz'))) as pbar:
    for name in os.listdir('xyz'):
        os.makedirs(os.path.join('xyz_label', name), exist_ok=True)
        for file in os.listdir(os.path.join('xyz', name)):
            if file.find('.xyz') == -1:
                continue
            xyz = np.loadtxt(os.path.join('xyz', name, file), dtype=str)
            # label = np.load(os.path.join('labels', name, file.replace('xyz', 'npy')), allow_pickle=True)
            # dist_local = label.item()['dist_local']
            retval2 = os.getcwd()
            residue_idx = -1
            current_residue = None
            for i in range(len(xyz)):
                residue = xyz[i, 0]
                if residue != current_residue:
                    residue_idx += 1
                    current_residue = residue
                aaa = xyz[i, 6]
                # xyz[i, 6] = dist_local[residue_idx].split()[-1]
            np.savetxt(os.path.join('xyz_label', name, file.replace('xyz', 'xyzlabel')), xyz, fmt='%s')
            pbar.set_description('%s/%s' % (name, file))
        pbar.update(1)
