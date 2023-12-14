
import os
import glob
import math
import torch
import pickle
import numpy as np
from tqdm import tqdm
from collections import Counter

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Bio import pairwise2

# aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
aa_list = ['G', 'A', 'V', 'L', 'I', 'M', 'C', 'S', 'T', 'N', 'Q', 'D', 'E', 'K', 'R', 'H', 'F', 'Y', 'W', 'P', 'X']
atom_list = ['N', 'C', 'O', 'S', 'H', 'X']
# atom_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
res_dict ={'GLY':'G','ALA':'A','VAL':'V','ILE':'I','LEU':'L','PHE':'F','PRO':'P','MET':'M','TRP':'W','CYS':'C',
               'SER':'S','THR':'T','ASN':'N','GLN':'Q','TYR':'Y','HIS':'H','ASP':'D','GLU':'E','LYS':'K','ARG':'R'}

Max_pssm = np.array([8, 9, 9, 9, 12, 10, 8, 8, 12, 9, 7, 9, 12, 10, 9, 8, 9, 13, 11, 8])
Min_pssm = np.array([-12, -12, -13, -13, -12, -11, -12, -12, -12, -12, -12, -12, -12, -12, -13, -12, -12, -13, -11, -12])
Max_hhm = np.array([12303, 12666, 12575, 12045, 12421, 12301, 12561, 12088, 12241, 11779, 12921, 12198, 12640, 12414, 12021, 11692, 11673, 12649, 12645, 12291])
Min_hhm = np.zeros(20)


fii = [-0.45982265,  0.72128421,  0.42540881, -0.14022461,  1.20850122,
        0.37162524, -0.50017869, -0.06248925,  0.91608461, -0.33312104,
       -0.82393211,  0.32918429,  0.86593674,  0.20117579,  0.01121351,
        0.131709  ,  0.06439939,  1.32943388,  0.51174203, -0.57461239,
        1.68613957]

class dssp():
    def process_dssp(self,dssp_file):
        aa_type = "ACDEFGHIKLMNPQRSTVWY"
        SS_type = "HBEGITSC"
        rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                    185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

        with open(dssp_file, "r") as f:
            lines = f.readlines()

        seq = ""
        dssp_feature = []

        p = 0
        while lines[p].strip()[0] != "#":
            p += 1
        for i in range(p + 1, len(lines)):
            aa = lines[i][13]
            if aa == "!" or aa == "*":
                continue
            seq += aa
            SS = lines[i][16]
            if SS == " ":
                SS = "C"
            SS_vec = np.zeros(9) # The last dim represents "Unknown" for missing residues
            SS_vec[SS_type.find(SS)] = 1
            PHI = float(lines[i][103:109].strip())
            PSI = float(lines[i][109:115].strip())
            ACC = float(lines[i][34:38].strip())
            ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
            dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

        return seq, dssp_feature


    def match_dssp(self,seq, dssp, ref_seq):
        alignments = pairwise2.align.globalxx(ref_seq, seq)
        ref_seq = alignments[0].seqA
        seq = alignments[0].seqB

        SS_vec = np.zeros(9) # The last dim represent "Unknown" for missing residues
        SS_vec[-1] = 1
        padded_item = np.concatenate((np.array([360, 360, 0]), SS_vec))

        new_dssp = []
        for aa in seq:
            if aa == "-":
                new_dssp.append(padded_item)
            else:
                new_dssp.append(dssp.pop(0))

        matched_dssp = []
        for i in range(len(ref_seq)):
            if ref_seq[i] == "-":
                continue
            matched_dssp.append(new_dssp[i])

        return matched_dssp


    def transform_dssp(self,dssp_feature):
        dssp_feature = np.array(dssp_feature)
        angle = dssp_feature[:,0:2]
        ASA_SS = dssp_feature[:,2:]

        radian = angle * (np.pi / 180)
        dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis = 1)

        return dssp_feature


    def get_dssp(self,data_path,pro_seq,filename):
        filename = filename[1:].strip('.')
        ref_seq = pro_seq[filename]
        dssp_seq, dssp_matrix = self.process_dssp(data_path)
        dssp_feature = []
        if dssp_seq != ref_seq:
            dssp_matrix = self.match_dssp(dssp_seq, dssp_matrix, ref_seq)
        dssp_feature = self.transform_dssp(dssp_matrix)

        return dssp_feature

class get_dataset(Dataset):
    def __init__(self, config, logger):
        self.logger = logger
        self.config = config
        self.debug = config.debug
        self.data_root = config.data_root
        self.dataset = config.dataset
        self.filename_suffix = config.filename_suffix
        self.test_ratio = config.test_ratio

        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.max_npoint = config.max_npoint

        self.voxel_size = config.voxel_size
        self.full_scale = config.full_scale

        self.all_files = []
        self.all_files_test129 = []
        self.all_files_test181 = []
        self.train_files = []
        self.val_files = []
        self.all_feature = {}
        self.fii = np.zeros(21)



    def process_hhm(self,hhm_file,file_name):
        # data = np.loadtxt(file, dtype=str)

        with open(hhm_file, "r") as f:
            lines = f.readlines()
        hhm_feature = []
        p = 0
        while lines[p][0] != "#":
            p += 1
        p += 5
        for i in range(p, len(lines), 3):
            if lines[i] == "//\n":
                continue
            feature = []
            record = lines[i].strip().split()[2:-1]
            for x in record:
                if x == "*":
                    feature.append(9999)
                else:
                    feature.append(int(x))
            hhm_feature.append(feature)
        hhm_feature = (np.array(hhm_feature) - Min_hhm) / (Max_hhm - Min_hhm)
        return hhm_feature


        
    def process_pssm(self , pssm_file):
        with open(pssm_file, "r") as f:
            lines = f.readlines()
        pssm_feature = []
        for line in lines:
            if line == "\n":
                continue
            record = line.strip().split()
            if record[0].isdigit():
                pssm_feature.append([int(x) for x in record[2:22]])
        pssm_feature = (np.array(pssm_feature) - Min_pssm) / (Max_pssm - Min_pssm)
        return pssm_feature

        return pssm_feature
    def pssm_hmm_dssp_feature(self):
        self.hmm_filename = 'hhm'
        feature_root = '/CrossBind/datasets/cath_decoys/DNA_Feature/feature'
        files = sorted(glob.glob(os.path.join(feature_root, '*/*%s' % self.hmm_filename)))

        num_files = len(files)

        if os.path.exists(os.path.join(self.data_root, 'feature%d.pickle' % num_files)):
            with open(os.path.join(self.data_root, 'feature%d.pickle' % num_files), 'rb')  as f:
                self.all_feature = pickle.load(f)
        else:
            self.logger.info('Process data (only in the first time)...')
            read_data,read_data_name = self.ProtDatasetFromCSV('/CrossBind/datasets/cath_decoys/DNA/DNA_ALL.fa')
            for f in tqdm(files, total=len(files)):
                ### f hhm root
                test_a = f.split('/')[-1]
                file_name = test_a.split('.')[0]
                if file_name == '4gfh_A':
                    continue
                if file_name in read_data:
                    gg = f.split('HMM')
                    pssm_f = gg[1].strip('hmm')
                    pssm_root = gg[0] + 'PSSM' + pssm_f + 'pssm'  
                    dssp_root = gg[0] + 'SS' + pssm_f + 'dssp'  
                    

                    hhm_feature = (self.process_hhm(f,file_name))
                    pssm_feature = self.process_pssm(pssm_root)
                    dssp_feature = dssp().get_dssp(dssp_root,read_data_name,pssm_f)


                    # animo_featre = np.hstack([hhm_feature, pssm_feature])
                    animo_featre = np.hstack([hhm_feature, pssm_feature,dssp_feature])


                    
                    self.all_feature.update({file_name: animo_featre})
                    # dssp_feature = self.
            with open(os.path.join(self.data_root, 'feature%d.pickle' % num_files), 'wb') as fi:
                pickle.dump(self.all_feature, fi)


    def init_pipeline(self):
        self.filename_suffix = 'xyz'
        # files = sorted(glob.glob(os.path.join(self.data_root, 'xyz_label', '*/*%s' % self.filename_suffix)))
        files = sorted(glob.glob(os.path.join(self.data_root, '*/*%s' % self.filename_suffix)))
        rooter = '/CrossBind/cfgs/'
        test_files = files = sorted(glob.glob(os.path.join(rooter, '*/*%s' % self.filename_suffix)))

        if self.config.get('mini_data_num', -1) != -1:
            files = files[:self.config.mini_data_num]

        if self.debug:
            files = files[:100]

        num_files = len(files)
        self.logger.info('Total samples are {}, using {} as validation samples...'.format(num_files, self.test_ratio))

        # if os.path.exists(os.path.join(self.data_root, 'preprare_data_948.pickle')):
        if os.path.exists(os.path.join(self.data_root, 'preprare_data%d.pickle' % num_files)):
            self.logger.info('Load pre-processed data...')

            with open(os.path.join(self.data_root, 'preprare_data%d.pickle' % num_files), 'rb') as f:
                self.all_files = pickle.load(f)


        else:
            self.logger.info('Process data (only in the first time)...')
            read_data,read_data_name = self.ProtDatasetFromCSV('/CrossBind/datasets/cath_decoys/DNA/DNA_Train_573.fa')

            propensity_all = np.zeros(21)
            for f in tqdm(files, total=len(files)):
            # for f in tqdm(test_files, total=len(test_files)):
                test_a = f.split('/')[-1]
                file_name = test_a.split('.')[0]

                if file_name in read_data:
                    self.all_files.append(self.load_data(f,read_data,read_data_name,propensity_all))
            aa_propensity = propensity_all / len(self.all_files)
            self.fii = np.log2(aa_propensity)

            # exit()
            # with open(os.path.join(self.data_root, 'preprare_data_948.pickle'), 'wb') as fi:
            with open(os.path.join(self.data_root, 'preprare_data_%d.pickle' % num_files), 'wb') as fi:'test_data_%d.pickle_181' 
                pickle.dump(self.all_files, fi)


        if os.path.exists(os.path.join(self.data_root, 'test%d.pickle' % num_files)):
            self.logger.info('Load pre-processed data...')

            with open(os.path.join(self.data_root, 'test_data_%d.pickle' % num_files), 'rb') as f:
            # with open(os.path.join(self.data_root, 'test_data_181.pickle'), 'rb') as f:
                self.all_files_test129 = pickle.load(f)
                a =1
            # with open(os.path.join(self.data_root, 'test_data_%d.pickle' % num_files), 'rb') as f:
            #     self.all_files_test181 = pickle.load(f)

        else:
            self.logger.info('Process data (only in the first time)...')
            # read_data_test,read_data_name_test = self.ProtDatasetFromCSV('/nvme/xusheng1/Linglin/resource/ProteinDecoy-main/datasets/cath_decoys/DNA/DNA_Test_181.fa')
            read_data_test,read_data_name_test = self.ProtDatasetFromCSV('/CrossBind/datasets/cath_decoys/DNA/DNA_Test_129.fa')
            propensity_all = np.zeros(21)
            # for f in tqdm(test_files, total=len(test_files)):
            for f in tqdm(files, total=len(files)):
                test_a = f.split('/')[-1]
                file_name = test_a.split('.')[0]

                if file_name in read_data_test:
                    self.all_files_test129.append(self.load_data(f,read_data_test,read_data_name_test,propensity_all))


            wwith open(os.path.join(self.data_root, 'test_data%d.pickle' % num_files), 'rb') as f:
                pickle.dump(self.all_files_test129, fi)

        self.train_files = self.all_files#[int(num_files * self.test_ratio):]
        self.val_files = self.all_files_test129#[:int(num_files * self.test_ratio)]
        # self.val_files = self.all_files[:int(num_files * self.test_ratio)]

    def openreadtxt(self,file_name):
        data_id = []
        data_seq = []
        data_label = []
        structure_emb = {}
        
        file = open(file_name,'r')  #打开文件
        file_data = file.readlines() #读取所有行

        for row in file_data:
            if row[0] == '>':
                # data_id.append(row[1:len(row)-1])
                data_id.append(row[1:].strip('\n'))

            elif (row[0].isdigit()):
                row = row.strip('\n')
                data_label.append(row)
            else:
                row = row.strip('\n')
                data_seq.append(row)

            
        return data_id,data_seq,data_label

    def ProtDatasetFromCSV(self,txt_path):
        data_dict = {}
        data_id_dna,data_seq_dna,data_label_dna = self.openreadtxt(txt_path)

        data_dict = dict(zip(data_id_dna,data_label_dna))
        data_dict_name = dict(zip(data_id_dna,data_seq_dna))
        return data_dict,data_dict_name

    def Load_feature(self,filename):
        feature = self.all_feature[filename]
        return feature

    def load_data(self, file,read_data,read_data_name,propensity_all):
        atom_number = []
        data = np.loadtxt(file, dtype=str)

        test_a = file.split('/')[-1]
        file_name = test_a.split('.')[0]
        target = read_data[file_name]
        amino_name = read_data_name[file_name]
        label = []
        for i in target:
            label.append(float(i))
        label_binary = np.array(label)
        aaa = Counter(label_binary)


        A_name = []
        for i in amino_name:
            A_name.append((i))
        AA_name = np.array(A_name)
        
        data_count = []
        clean_data = []
        for i in range(len(data)):
            m = 0
            a = data[i]
            c = a[1]
            if c == '':
                c = a[1]
            if c[-1] == '!':
                m = 1
                continue
            b = a[0]
            if_word = b[-1].isdigit()
            if if_word:
                clean_data.append(data[i,:])
                data_count.append(b)
        
        # data_nox = np.concatenate(clean_data,0)
        data_nox = np.array(clean_data)
        # s_l = len(data_count)
        # data_nox = data[:s_l,:]
        xyz = data_nox[:, 2:5].astype(float)
        label = data_nox[:, -1].astype(float)
        aa_feature = torch.zeros(label.shape[0], 21)
        atom_feature = torch.zeros(label.shape[0], 6)
        residue_idx = data_nox[:, 0]


        ########  AA_propensity
        aa_array = np.array(aa_list)
        asas = np.append(aa_array,AA_name,0)
        AA_propensity = dict(Counter(asas))
        


    
        idx_p = np.where(label_binary == 1)
        binding_AA = AA_name[idx_p]
        binding_AA = binding_AA
        adada = np.append(aa_array,binding_AA,0)
        # binding_AA.extend(aa_array.squeeze(0))
        AA_binding = dict(Counter(adada))

        for k ,v in AA_propensity.items():
            if AA_propensity[k] - AA_binding[k] != 0:
                AA_propensity[k] = AA_propensity[k] - AA_binding[k]
        a = {k: v / total for total in (sum(AA_propensity.values()),) for k, v in AA_propensity.items()}
        b = {k: v / total for total in (sum(AA_binding.values()),) for k, v in AA_binding.items()}
        c = a.copy()
        i = 0
        for k ,v in a.items():
            c[k] = (float(b[k]/a[k]))
            propensity_all[i] += c[k]
            i += 1

        index = 0
        new_data = []
        ind = 0
        result = dict(Counter(data_count))
        result_new = result.copy()
        for key,value in result.items():
            if value < 3:
                print(value)
                del result_new[key]

        print(len(label_binary) , len(result_new))

        if len(label_binary) != len(result_new):
            test = 1
        ccc = label.shape
        atom_count = []
        for i in result_new.values():

            # print(len(label_binary) , ind)
            label[index:(index+i)] = label_binary[ind]
            ind += 1
            index = i + index
            atom_count.append(i)

        if index != label.shape[0]:
            print(result_new)
            print(index,label.shape)
            print(file_name)
        # print('aaa', aaa)
        # bbbb = Counter(label)

        atom_name = []
        for i, feat in enumerate(data_nox[:, 1]):

            if feat[-1] == '!':
                print(feat)
                continue
            # print(feat)
            aa_feature[i, aa_list.index(feat[0])] = 1
            atom_feature[i, atom_list.index(feat[1])] = 1
            atom_name.append(feat)  



        index = 0
        aat = []
        j = 0
        for z in (atom_count):
            # idn += 1
            # aaab = result_new[j]
            aat.append(atom_name[index:(index+z)])
            
            asst = aat[j]
            vc = asst[-1]
            ccc = vc[0]
            if AA_name[j] != ccc:
                abc = list(result_new.keys())[j]
                print(file_name,AA_name[j] , aat[j])
                print('11')
            # print(AA_name[index:(index+z)])
            # aa_feature[id,] = AA_name[index:(index+i),:]
            index += z
            j += 1

        return {
            'xyz': xyz,
            'aa_feature': aa_feature,
            'atom_feature': atom_feature,
            'label': label,
            'label_binary': label_binary,
            'residue_idx': residue_idx,
            'file': file,
            'file_name': file_name,
            'protein_seq': amino_name,
            'atom_count': atom_count,
            'atom_name': atom_name,
            'amino':  AA_name,
            'index_num': result_new,
            'propensity_all':propensity_all
        }   

    def init_trainloader(self):
        self.logger.info('Training samples: {}'.format(len(self.train_files)))

        train_set = list(range(len(self.train_files)))
        self.train_data_loader = DataLoader(
            train_set, 
            batch_size=self.batch_size,
            collate_fn=self.trainMerge,
            num_workers=self.num_workers,
            shuffle=True,
            sampler=None,
            drop_last=True,
            pin_memory=True)

    def init_valloader(self):
        self.logger.info('Validation samples: {}'.format(len(self.val_files)))

        val_set = list(range(len(self.val_files)))
        self.val_data_loader = DataLoader(
            val_set, 
            batch_size=1,
            collate_fn=self.valMerge,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True)

    def process_input_data(self, data_dict, test=False):
        
        xyz = data_dict['xyz']
        feature = np.concatenate([data_dict['aa_feature'], data_dict['atom_feature']], 1)

        # feature = data_dict['aa_feature']
        label = data_dict['label']
        atom_num = data_dict['atom_count']
        aaa = Counter(label)
        file_name = data_dict['file_name']
        if file_name == '4gfh_A':
            test = 1
        label_binary = data_dict['label_binary']

        if not test:
        # divide by voxel size
            ciii = (xyz - xyz.mean(0))
            coords = np.ascontiguousarray(xyz - xyz.mean(0))
            m = np.eye(3) + np.random.randn(3, 3) * 0.1
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
            m /= self.voxel_size

            # rotation (currently only on z-axix)
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
            coords = np.matmul(coords, m)

            # place on (0,0) and crop out the voxel outside the full_scale
            m = coords.min(0)
            M = coords.max(0)
            offset = - m + np.clip(self.full_scale - M + m - 0.001, 0, None) * np.random.rand(3) + \
                    np.clip(self.full_scale - M + m + 0.001, None, 0) * np.random.rand(3)

            coords += offset

        if not test:
            idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)
            coords = coords[idxs]
            feature = feature[idxs]
            label = label[idxs]
            # atom_num = atom_num[idxs]

        if test:
            # coords = xyz
            coords = np.ascontiguousarray(xyz - xyz.mean(0))
            m = np.eye(3) 
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
            m /= self.voxel_size

            theta = np.random.rand() * 2 * math.pi
            m= np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
            # coords = np.matmul(coords, m)
            coords=np.matmul(coords,m)+4096/2+np.random.uniform(-2,2,3)

            m = coords.min(0)
            M = coords.max(0)

            offset = - m + np.clip(self.full_scale - M + m - 0.001, 0, None) * np.random.rand(3) + \
                    np.clip(self.full_scale - M + m + 0.001, None, 0) * np.random.rand(3)

            coords += offset

            # coords = np.ascontiguousarray(xyz - xyz.mean(0))
            # m = np.eye(3) + np.random.randn(3, 3) * 0.1
            # m[0][0] *= np.random.randint(0, 2) * 2 - 1
            # m /= self.voxel_size

            # # rotation (currently only on z-axix)
            # theta = np.random.rand() * 2 * math.pi
            # m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
            # coords = np.matmul(coords, m)

            # # place on (0,0) and crop out the voxel outside the full_scale
            # m = coords.min(0)
            # M = coords.max(0)
            # offset = - m + np.clip(self.full_scale - M + m - 0.001, 0, None) * np.random.rand(3) + \
            #         np.clip(self.full_scale - M + m + 0.001, None, 0) * np.random.rand(3)

            # coords += offset

            # idxs=(coords.min(1)>=0)*(coords.max(1)<4096)
            # idxs = np.where(coords<0) 
            # idxs2 = np.where(coords>4096)
            # coords[idxs] = 0 
            # coords[idxs2] = 0


        coords = torch.Tensor(coords).long()
        feature = torch.Tensor(feature)
        label = torch.Tensor(label)

        atana = data_dict['atom_name']
        A_name = []
        for i in atana:
            A_name.append((i))
        AA_name = np.array(A_name)
        idn = 0

        
        amino = data_dict['amino']
        # for id in range(amino.shape[0]):
        index = 0
        aat = []
        j = 0
        for z in (atom_num):
            idn += 1
            aa_feature = torch.zeros(amino.shape[0] ,14)       

            aat.append(AA_name[index:(index+z)])
            # print(file_name,amino[j] , aat[j])
            bbb = amino[j]
            asst = aat[j]
            vc = asst[-2]
            ccc = vc[0]
            if amino[j] != ccc:
                print(file_name,amino[j] , aat[j])
                print('11')
            # print(AA_name[index:(index+z)])
            # aa_feature[id,] = AA_name[index:(index+i),:]
            index += z
            j += 1
                # print('1')



        return coords, label, feature, data_dict['file'] , file_name , atom_num,label_binary, data_dict['protein_seq']

    def trainMerge(self, indices):
        coords = []
        features = []
        labels = []
        file_name_list = []
        atom_number = []
        label_binarys = []
        peotein_seq = []
        for idx, index in enumerate(indices):
            coord, label, feature , filename ,file_name,atom_num,label_binary,sequence = self.process_input_data(self.all_files[index],test = False)
            c = np.where(label == 1.0)
            coords.append(torch.cat([coord, torch.LongTensor(coord.shape[0], 1).fill_(idx)], 1))
            labels.append(label)
            features.append(feature)
            file_name_list.append(file_name)
            atom_number.append(atom_num)
            label_binarys.append(label_binary)
            peotein_seq.append(sequence)

        c = np.where(labels == 1.0)
        # coords = torch.cat(coords, 0)
        inputs = {
            'coords': torch.cat(coords, 0),
            'reg_labels': torch.cat(labels, 0),
            'label_binary': label_binarys,
            'features': torch.cat(features, 0),
            'file_name': file_name_list,
            'atom_num': atom_number,
            'protein_seq': peotein_seq,
            
        }

        if len(inputs['coords']) > self.config.max_npoint:
            sample_idx = np.random.sample(len(inputs['coords']), self.config.max_npoint)
            inputs['coords'] = inputs['coords'][sample_idx]
            inputs['reg_labels'] = inputs['reg_labels'][sample_idx]
            inputs['features'] = inputs['features'][sample_idx]

        if 'classification' in self.config:
            inputs['cls_labels'] = inputs['reg_labels'] * self.config.classification.num_bins
            inputs['cls_labels'] = inputs['cls_labels'].long()

        return inputs

    def valMerge(self, indices):
        coords = []
        features = []
        labels = []
        file_name_list = []
        atom_number = []
        label_binarys = []
        peotein_seq = []
        for idx, index in enumerate(indices):
            coord, label, feature , filename ,file_name,atom_num,label_binary, sequence = self.process_input_data(self.all_files_test129[index],test=True)

            # filenames.append(filename)
            # coords.append(torch.cat([coord, torch.LongTensor(coord.shape[0], 1).fill_(idx)], 1))
            # labels.append(label)
            # features.append(feature)

            c = np.where(label == 1.0)
            coords.append(torch.cat([coord, torch.LongTensor(coord.shape[0], 1).fill_(idx)], 1))
            labels.append(label)
            features.append(feature)
            file_name_list.append(file_name)
            atom_number.append(atom_num)
            label_binarys.append(label_binary)
            peotein_seq.append(sequence)
        inputs = {
            'coords': torch.cat(coords, 0),
            'reg_labels': torch.cat(labels, 0),
            'label_binary': label_binarys,
            'features': torch.cat(features, 0),
            'file_name': file_name_list,
            'atom_num': atom_number,
            'protein_seq': peotein_seq
        }

        if 'classification' in self.config:
            inputs['cls_labels'] = inputs['reg_labels'] * self.config.classification.num_bins
            inputs['cls_labels'] = inputs['cls_labels'].long()

        return inputs
