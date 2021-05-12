import os
import sys
import glob
import copy
import numpy as np
from biopandas.pdb import PandasPdb
from keras.utils import np_utils
from collections import Counter

# ----------------------------------------------------------------------------
res_order = ['GLY','ALA','VAL','LEU','ILE','MET','PHE','TRP','PRO','SER','THR','CYS','TYR','ASN','GLN','ASP','GLU','LYS','ARG','HIS']
his_alter = ['HIS', 'HIE', 'HID']
negative_res = ['ASP', 'GLU']
positive_res = ['LYS', 'ARG']
sb_cutoff = 3.9

# ----------------------------------------------------------------------------
class Featurizer:
    def __init__(self, data_dir, work_dir, error_log):
        self.data_dir = data_dir
        self.work_dir = work_dir    # 데이터 저장할 디렉토리 (trans_dir)

        self.error_log = error_log

    ###########################################################################
    # residual solvation free energy
    # xmu_res(ires), xmu_res_LJ(ires), xmu_res_cou(ires)
    def residual_Gsolv(self, pdb_code, xmu_dir='xmu', pdb_dir='pdb',):
        target_file1 = '{}/ydata/{}_RGsolv.npy'.format(self.work_dir, pdb_code)
        target_file2 = '{}/ydata/{}_RGsolv_LJ.npy'.format(self.work_dir, pdb_code)
        target_file3 = '{}/ydata/{}_RGsolv_elec.npy'.format(self.work_dir, pdb_code)
        if os.path.exists(target_file3):
            return 'already exist'

        f_list = sorted(glob.glob('{}/{}/xmu_res_*.dat'.format(self.data_dir, xmu_dir)))
        f_list2 = sorted(glob.glob('{}/{}/*.pdb'.format(self.data_dir, pdb_dir)))

        residue_num = len(f_list)  # no of residue
        data_num = len(f_list2)    # no of conformation

        Gsolv = np.empty((data_num,0))
        Gsolv_LJ = np.empty((data_num,0))
        Gsolv_elec = np.empty((data_num,0))

        for j in range(residue_num):
            xmu = np.loadtxt(f_list[j])[:,1].reshape(-1,1)
            Gsolv = np.concatenate((Gsolv, xmu), axis=1)

            xmu_LJ = np.loadtxt(f_list[j])[:,2].reshape(-1,1)
            Gsolv_LJ = np.concatenate((Gsolv_LJ, xmu_LJ), axis=1)

            xmu_elec = np.loadtxt(f_list[j])[:,3].reshape(-1,1)
            Gsolv_elec = np.concatenate((Gsolv_elec, xmu_elec), axis=1)
            # print('Gsolv :', Gsolv.shape)

        np.save(target_file1, Gsolv)
        np.save(target_file2, Gsolv_LJ)
        np.save(target_file3, Gsolv_elec)
        # print(Gsolv)

        return Gsolv.shape


    def residual_Gsolv_slice(self, pdb_code, xmu_dir='xmu', pdb_dir='pdb', slice=1):
        f_list = sorted(glob.glob('{}/{}/xmu_res_*.dat'.format(self.data_dir, xmu_dir)))
        f_list2 = sorted(glob.glob('{}/{}/*.pdb'.format(self.data_dir, pdb_dir)))

        residue_num = len(f_list)  # no of residue
        data_num = len(f_list2)    # no of conformation

        subset_num = int(data_num/slice)
        print('subset_num :', subset_num)

        for part_no in range(slice):
            start_idx = part_no * subset_num
            end_idx = start_idx + subset_num
            print(start_idx, end_idx)

            target_file1 = '{}/ydata/{}{}_RGsolv.npy'.format(self.work_dir, pdb_code, part_no+1)
            target_file2 = '{}/ydata/{}{}_RGsolv_LJ.npy'.format(self.work_dir, pdb_code, part_no+1)
            target_file3 = '{}/ydata/{}{}_RGsolv_elec.npy'.format(self.work_dir, pdb_code, part_no+1)
            if os.path.exists(target_file3):
                continue

            Gsolv = np.empty((subset_num,0))
            Gsolv_LJ = np.empty((subset_num,0))
            Gsolv_elec = np.empty((subset_num,0))

            for j in range(residue_num):
                xmu = np.loadtxt(f_list[j])[start_idx:end_idx,1].reshape(-1,1)
                xmu_LJ = np.loadtxt(f_list[j])[start_idx:end_idx,2].reshape(-1,1)
                xmu_elec = np.loadtxt(f_list[j])[start_idx:end_idx,3].reshape(-1,1)

                Gsolv = np.concatenate((Gsolv, xmu), axis=1)
                Gsolv_LJ = np.concatenate((Gsolv_LJ, xmu_LJ), axis=1)
                Gsolv_elec = np.concatenate((Gsolv_elec, xmu_elec), axis=1)

            np.save(target_file1, Gsolv)
            np.save(target_file2, Gsolv_LJ)
            np.save(target_file3, Gsolv_elec)
            print('Gsolv :', Gsolv.shape)

        return data_num

    ###########################################################################
    # total charge
    def calc_netcharge(self, sequence):
        net_charge = 0

        res_neg = [x for x in sequence if x in negative_res]
        res_pos = [x for x in sequence if x in positive_res]
        net_charge = len(res_pos) - len(res_neg)

        return net_charge

    def total_charge(self, pdb_code, conformation_num, res_num, res_sequence):
        target_file = '{}/charge/{}_charge.npy'.format(self.work_dir, pdb_code)
        if os.path.exists(target_file):
            return 'already exist'

        net_charge = self.calc_netcharge(res_sequence)
        print('net charge of {} is {}'.format(pdb_code, net_charge))

        temp_charge = []
        for i in range(conformation_num):    # no of conformation
            for j in range(res_num):
                temp_charge.append(net_charge)
        temp_charge = np.array(temp_charge).reshape(-1,1)

        np.save(target_file, temp_charge)

        return temp_charge.shape

    def total_charge_slice(self, pdb_code, conformation_num, res_num, res_sequence, slice=1):
        net_charge = self.calc_netcharge(res_sequence)
        print('net charge of {} is {}'.format(pdb_code, net_charge))

        subset_num = int(conformation_num/slice)
        print('subset_num :', subset_num)

        for part_no in range(slice):
            target_file = '{}/charge/{}{}_charge.npy'.format(self.work_dir, pdb_code, part_no+1)
            if os.path.exists(target_file):
                continue

            temp_charge = []
            for i in range(subset_num):
                for j in range(res_num):
                    temp_charge.append(net_charge)
            temp_charge = np.array(temp_charge).reshape(-1,1)

            np.save(target_file, temp_charge)

        return subset_num * slice


    ###########################################################################
    # residue type - one hot
    def res_type_list(self, res_seq, neighbor=''):
        temp_seq = []

        if neighbor == 'prev':
            temp_seq.append(20)

        for res_name in res_seq:
            if res_name=='GLY':temp_rtype=0
            elif res_name=='ALA':temp_rtype=1
            elif res_name=='VAL':temp_rtype=2
            elif res_name=='LEU':temp_rtype=3
            elif res_name=='ILE':temp_rtype=4
            elif res_name=='MET':temp_rtype=5
            elif res_name=='PHE':temp_rtype=6
            elif res_name=='TRP':temp_rtype=7
            elif res_name=='PRO':temp_rtype=8
            elif res_name=='SER':temp_rtype=9
            elif res_name=='THR':temp_rtype=10
            elif res_name in ['CYS','CYX']:temp_rtype=11
            elif res_name=='TYR':temp_rtype=12
            elif res_name=='ASN':temp_rtype=13
            elif res_name=='GLN':temp_rtype=14
            elif res_name=='ASP':temp_rtype=15
            elif res_name=='GLU':temp_rtype=16
            elif res_name=='LYS':temp_rtype=17
            elif res_name=='ARG':temp_rtype=18
            elif res_name in ['HIS','HID','HIE','HIP']:temp_rtype=19
            else:temp_rtype=20  # None

            temp_seq.append(temp_rtype)

        if neighbor == 'next':
            temp_seq.append(20)

        return temp_seq

    def residue_type(self, pdb_code, conformation_num, res_num, res_sequence):
        target_file = '{}/RT/{}_RT_curr.npy'.format(self.work_dir, pdb_code)
        if os.path.exists(target_file):
            return 'already exist'

        res_now = self.res_type_list(res_sequence)
        RT = np.array([res_now]*conformation_num).reshape(-1,1)
        RT_onehot = np_utils.to_categorical(RT, 21)

        np.save(target_file, RT_onehot)

        return RT_onehot.shape

    def residue_type_slice(self, pdb_code, conformation_num, res_num, res_sequence, slice=1):
        subset_num = int(conformation_num/slice)
        print('subset_num :', subset_num)

        for part_no in range(slice):
            target_file = '{}/RT/{}{}_RT_curr.npy'.format(self.work_dir, pdb_code, part_no+1)
            if os.path.exists(target_file):
                continue

            res_now = self.res_type_list(res_sequence)
            RT = np.array([res_now]*subset_num).reshape(-1,1)
            RT_onehot = np_utils.to_categorical(RT, 21)
            print(RT_onehot.shape)

            np.save(target_file, RT_onehot)

        return conformation_num


    ###########################################################################
    # fraction of reside type
    def residue_fraction(self, pdb_code, conformation_num, res_num, res_sequence):
        target_file = '{}/RT_frac/{}_RT_frac.npy'.format(self.work_dir, pdb_code)
        if os.path.exists(target_file):
            return 'already exist'

        cat_res = Counter(res_sequence)
        # print(cat_res)

        if cat_res['HIE'] > 0:
            cat_res['HIS'] = cat_res.pop('HIE')
        if cat_res['HID'] > 0:
            cat_res['HIS'] = cat_res.pop('HID')

        if cat_res['CYX'] > 0:
            if cat_res['CYS'] > 0:
                cat_res['CYS'] = cat_res['CYS'] + cat_res['CYX']
                cat_res.pop('CYX')
            else:
                cat_res['CYS'] = cat_res.pop('CYX')

        r_sum = 0
        r_nega_sum = 0
        r_posi_sum = 0

        # AA 순서대로 count. charged residue도 count.
        for r_name in res_order:
            r_cnt = cat_res[r_name]
            r_sum += r_cnt

            if r_name in negative_res:
                r_nega_sum += r_cnt

            if r_name in positive_res:
                r_posi_sum += r_cnt
            # 갯수를 비율로 변환
            cat_res[r_name] = float('{0:.5f}'.format(r_cnt/res_num))

        if res_num != r_sum:
            print('{} --- different residue count : {} - {}'.format(pdb_code, res_num, r_sum))
            with open(self.error_log, "a") as f:
                print('{} --- different residue count : {} - {}'.format(pdb_code, res_num, r_sum), file=f)

        res_frac = []
        for r_name in res_order:
            res_frac.append(cat_res[r_name])

        res_frac.append(float('{0:.5f}'.format(r_nega_sum/res_num)))
        res_frac.append(float('{0:.5f}'.format(r_posi_sum/res_num)))
        # print(res_frac)

        res_frac_all = np.array([res_frac] * res_num * conformation_num)

        np.save(target_file, res_frac_all)

        return res_frac_all.shape


    ###########################################################################
    # number of residues
    def total_residue(self, pdb_code, conformation_num, res_num):
        target_file = '{}/residue/{}_total_residue.npy'.format(self.work_dir, pdb_code)
        if os.path.exists(target_file):
            return 'already exist'

        temp_res = []
        for i in range(conformation_num):
            for j in range(res_num):
                temp_res.append(res_num)
        temp_res = np.array(temp_res).reshape(-1,1)

        np.save(target_file, temp_res)

        return temp_res.shape


    ###########################################################################
    # N terminal = -1
    # C terminal = 1
    def terminal_type(self, pdb_code, conformation_num, res_num):
        target_file = '{}/terminal/{}_terminal.npy'.format(self.work_dir, pdb_code)
        if os.path.exists(target_file):
            return 'already exist'

        temp_terminal = []
        for i in range(conformation_num):    # conformation 개수
            temp_terminal.append(-1)
            for k in range(res_num-2):
                temp_terminal.append(0)
            temp_terminal.append(1)
            # print(temp_terminal)

        temp_terminal = np.array(temp_terminal).reshape(-1,1)

        np.save(target_file, temp_terminal)

        return temp_terminal.shape


    ###########################################################################
    # salt bridge
    """
    (-) ASP, GLU    : 2개의 O - OD1,OD2 / OE1,OE2
    (+) LYS, ARG    : N에 있는 H - HZ1,HZ2,HZ3 / HH11,HH12,HH21,HH22
    """
    def cal_distance(self, df1, df2):
        rnum1 = df1.shape[0]
        rnum2 = df2.shape[0]

        # distance_list = []
        min_dist = 4.0
        min_idx = 9999
        for i in range(rnum1):
            for j in range(rnum2):
                x1 = np.array(df1.iloc[i])
                x2 = np.array(df2.iloc[j])
                x = x1 - x2
                distance = np.sqrt(np.dot(x,x))
                if distance < min_dist:
                    min_dist = distance
                    min_idx = j
                # if distance > 0:
                #     distance_list.append(distance)

        # if min_idx < 9999:
        #     print(min_dist)
        #     print(min_idx)
        #     print(df2.iloc[min_idx])

        return min_dist, min_idx
        # return np.min(distance_list)

    def check_salt_bridge(self, charge, ridx, comp_list, df_org):
        sb_yesno = 0
        sb_res = 0
        dist = 0

        compare_list = copy.copy(comp_list)

        # neighbor residue 제외
        if (ridx-1) in compare_list:
            compare_list.remove(ridx-1)
        if (ridx-2) in compare_list:
            compare_list.remove(ridx-2)
        if (ridx+1) in compare_list:
            compare_list.remove(ridx+1)
        if (ridx+2) in compare_list:
            compare_list.remove(ridx+2)

        if len(compare_list) == 0:
            sb_yesno = 0
        else:
            if charge == 'neg':
                df_res = df_org.loc[(df_org["residue_number"] == ridx+1), ['residue_name','residue_number','atom_name','x_coord','y_coord','z_coord']]
                df_res = df_res[df_res['atom_name'].isin(['OD1','OD2','OE1','OE2'])]

                c_list = [x+1 for x in compare_list]
                df_com_org = df_org.loc[df_org['residue_number'].isin(c_list), ['residue_name','residue_number','atom_name','x_coord','y_coord','z_coord']]
                df_com_org = df_com_org[df_com_org['atom_name'].isin(['HZ1','HZ2','HZ3','HH11','HH12','HH21','HH22'])]

                df_res = df_res.loc[:,['x_coord','y_coord','z_coord']]
                df_com = df_com_org.loc[:,['x_coord','y_coord','z_coord']]
                dist, dist_idx = self.cal_distance(df_res, df_com)

            else:   # pos
                df_res = df_org.loc[(df_org['residue_number'] == ridx+1), ['residue_name','residue_number','atom_name','x_coord','y_coord','z_coord']]
                df_res = df_res[df_res['atom_name'].isin(['HZ1','HZ2','HZ3','HH11','HH12','HH21','HH22'])]

                c_list = [x+1 for x in compare_list]
                df_com_org = df_org.loc[df_org['residue_number'].isin(c_list), ['residue_name','residue_number','atom_name','x_coord','y_coord','z_coord']]
                df_com_org = df_com_org[df_com_org['atom_name'].isin(['OD1','OD2','OE1','OE2'])]

                df_res = df_res.loc[:,['x_coord','y_coord','z_coord']]
                df_com = df_com_org.loc[:,['x_coord','y_coord','z_coord']]
                dist, dist_idx = self.cal_distance(df_res, df_com)

            if dist < sb_cutoff:
                # print('dist :', dist)
                # print('dist_idx :', dist_idx)
                # print(df_com_org.iloc[dist_idx, :])
                sb_res = (df_com_org.iloc[dist_idx, :]['residue_number']) - 1
                sb_yesno = 1

        return sb_yesno, sb_res

    def salt_bridge(self, pdb_code, res_num, res_sequence):
        target_file = '{}/saltbridge/{}_saltbridge.npy'.format(self.work_dir, pdb_code)
        if os.path.exists(target_file):
            return 'already exist'

        # 1개 protein 전체 20개 구조의 salt bridge yes/no
        protein_sb_list = []

        # charged residue 리스트 작성
        neg_list = []
        pos_list = []
        for k, res in zip(range(res_num), res_sequence):
            if res in negative_res:
                neg_list.append(k)
            elif res in positive_res:
                pos_list.append(k)
            else:
                pass

        # print(res_num)
        # print(neg_list)
        # print(pos_list)

        # process all files
        f_list = sorted(glob.glob('{}/pdb/*.pdb'.format(self.data_dir)))
        data_num = len(f_list)

        for i in range(data_num):
            ppdb = PandasPdb()
            ppdb.read_pdb(f_list[i])
            df = ppdb.df['ATOM'][['residue_number','residue_name','atom_name','x_coord','y_coord','z_coord','chain_id']]
            df = df[df['chain_id'] == 'A']

            # OpenMM에서 pdb 생성이 잘못된 경우
            if df.shape[0] == 0:
                return 'error_pdb'

            # check salt bridge
            sb_res_list = []
            sb_entry_list = []
            sb_exist = 0
            if len(neg_list) == 0 or len(pos_list) == 0:  # salt_bridge 없음
                sb_res_list = [0] * res_num
            else:
                for k in range(res_num):
                    if k in sb_entry_list:    # 이미 saltbridge 하고 있는 것으로 체크된 것은 skip
                        sb_exist = 1
                    else:
                        if k in neg_list:
                            sb_exist, sb_partner = self.check_salt_bridge('neg', k, pos_list, df)
                            if sb_exist == 1:
                                sb_entry_list.append(k)
                                sb_entry_list.append(sb_partner)
                        elif k in pos_list:
                            sb_exist, sb_partner = self.check_salt_bridge('pos', k, neg_list, df)
                            if sb_exist == 1:
                                sb_entry_list.append(k)
                                sb_entry_list.append(sb_partner)
                        else:
                            sb_exist = 0

                    sb_res_list.append(sb_exist)

            # print(sb_res_list)
            protein_sb_list = np.concatenate([protein_sb_list, sb_res_list])

            print(os.path.basename(f_list[i])[:10])

        sb_array = np.array(protein_sb_list).reshape(len(protein_sb_list), 1)

        np.save(target_file, sb_array)

        return sb_array.shape


    ###########################################################################
