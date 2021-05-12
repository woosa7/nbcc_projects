import glob
import os
import math
import subprocess
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
import sys


class Image3D_LJ_elec:
    def __init__(self, src_dir, work_dir, pdb_code, error_log):
        self.source_dir = src_dir   # md_simulation/1A/1A00A
        self.work_dir = work_dir    # trans_dir
        self.pdb_code = pdb_code
        self.error_log = error_log

    #################################### distance calculate ######################################
    def cal_distance(self, rn, center, df, cutoff):            # 비교할 residue번호, 비교할 residue의 CB좌표, 모든 좌표
        AT = len(df)   # 총 atom의 개수
        center = center.reshape(-1)

        data=[]
        for i in range(AT):                    # 모든 원자와 좌표 계산
            if abs(rn+1-df[i,0]) > 0:          # 비교대상 residue와 앞뒤로 붙어있는 2개의 residue는 거리계산에서 제외 하고 싶으면 >0 을 >2로 바꾸면됨
                x = center-df[i,3:6]           # 비교대상 CB - 다른 atom
                xsqrt = np.dot(x,x)            # 제곱
                distance = np.sqrt(xsqrt)      # 루트
                if distance > 0 and distance <= cutoff:  # cutoff 안쪽 atom만 담아놓기
                    data.append(df[i])
        data = np.array(data)
        return data  # residue번호, residue이름, atom이름, X, Y, Z 의 정보를 담고있는 배열 (N , 6)

    ######################################## 원점 이동  ###########################################
    def coord_move(self, center, In_residue, Out_residue, ATN, OTN):
        for i in range(ATN):
            In_residue[i,2:5] = In_residue[i,2:5]-center        # residue에 해당하는 ATOM좌표 - CB좌표 -> CB를 원점으로 이동
        for j in range(OTN):
            Out_residue[j,3:6] = Out_residue[j,3:6]-center      # residue외에 해당하는 ATOM좌표 - CB좌표 -> CB를 원점으로 이동

        return In_residue, Out_residue

    ##################################### rotating matrix ########################################
    def rotating_function(self, axis, Q, temp):
        if axis == 1:  # Z축 회전
            M = np.array([[math.cos(Q), math.sin(Q), 0],[-math.sin(Q), math.cos(Q), 0],[0,0,1]])
            CASE = abs(np.dot(temp,M)[0])
        elif axis == 2:  # X축 회전
            M = np.array([[1,0,0],[0, math.cos(Q), math.sin(Q)],[0,-math.sin(Q),math.cos(Q)]])
            CASE = abs(np.dot(temp,M)[2])
        elif axis == 3:  # Y축 회전
            M = np.array([[math.cos(Q),0,-math.sin(Q)],[0, 1, 0],[math.sin(Q),0,math.cos(Q)]])
            CASE = abs(np.dot(temp,M)[2])
        return M, CASE

    #################################### rotating calculate ######################################
    def cal_rotating(self, CA, N, C, axis, In_residue, Out_residue, T_CB, ATN, OTN):
        if axis == 1:      # N 좌표를 YZ 평면으로 Z축 회전
            v1 = np.array([0,1,0])
            v2 = np.array([N[0],N[1],0])
            temp = N
        elif axis == 2:   # N 좌표를 XY 평면으로 X축 회전
            v1 = np.array([0,1,0])
            v2 = np.array([0,N[1],N[2]])
            temp = N
        elif axis == 3:   # C 좌표를 XY 평면으로 Y축 회전
            v1 = np.array([1,0,0])
            v2 = np.array([C[0],0,C[2]])
            temp = C

        v1_d = np.sqrt(np.sum(np.square(v1)))
        v2_d = np.sqrt(np.sum(np.square(v2)))

        Q = math.acos((v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) / (v1_d * v2_d))
        M,CASE1 = self.rotating_function(axis,Q,temp)

        Q = -math.acos((v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) / (v1_d * v2_d))
        M,CASE2 = self.rotating_function(axis,Q,temp)

        if CASE1<=CASE2:
            Q = math.acos((v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) / (v1_d * v2_d))
            M,_ = self.rotating_function(axis,Q,temp)
            CA = np.dot(CA,M)
            N = np.dot(N,M)
            C = np.dot(C,M)
            T_CB = np.dot(T_CB,M)

            for i_1 in range(ATN):          # residue에 해당 모든 좌표 회전
                In_residue[i_1,2:5] = np.dot(In_residue[i_1,2:5],M)
            for i_2 in range(OTN):          # 반경내의 모든 좌표도 회전
                Out_residue[i_2,3:6] = np.dot(Out_residue[i_2,3:6],M)
        else :
            Q = -math.acos((v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) / (v1_d * v2_d))
            M,_ = self.rotating_function(axis,Q,temp)
            CA = np.dot(CA,M)
            N = np.dot(N,M)
            C = np.dot(C,M)
            T_CB = np.dot(T_CB,M)

            for i_1 in range(ATN):          # residue에 해당 모든 좌표 회전
                In_residue[i_1,2:5] = np.dot(In_residue[i_1,2:5],M)
            for i_2 in range(OTN):          # 반경내의 모든 좌표도 회전
                Out_residue[i_2,3:6] = np.dot(Out_residue[i_2,3:6],M)

        return CA, N, C, In_residue, Out_residue, T_CB

    ##############################################################################################
    # Main function
    def make_feature(self, xmu_dir='xmu', pdb_dir='pdb', slice=1):
        ppdb = PandasPdb()

        # target_file1 = '{}/3D/{}_3D_atom_charge.npy'.format(self.work_dir, self.pdb_code)
        # target_file2 = '{}/3D/{}_3D_LJ_sigma.npy'.format(self.work_dir, self.pdb_code)
        # target_file3 = '{}/3D/{}_3D_LJ_epsilon.npy'.format(self.work_dir, self.pdb_code)
        # if os.path.exists(target_file3):
        #     return 'already exist'

        # protein.pql
        pql_file = '{}/protein.pql'.format(self.source_dir)
        if not os.path.exists(pql_file):
            err_msg = 'not exist --- {}'.format(pql_file)
            with open(self.error_log, "a") as f:
                print(err_msg, file=f)
            return err_msg

        col_names = ['ATOM','atom_number','atom_name','residue_name','residue_number','atom_charge','LJ_sigma','LJ_epsilon']
        df_pql = pd.read_csv(pql_file, sep='\s+', skiprows=1, names = col_names)
        # df_pql.loc[(df_pql.residue_number == 1) & (df_pql.atom_name == 'H1'), 'atom_name'] = 'H'  # N-terminal : H1 --> H
        # df_pql = df_pql.drop(df_pql[(df_pql.residue_name  == 'CYS') & (df_pql.atom_name == 'HG')].index) # CYS HG 삭제 - OpenMM pdb에 없음.
        df_pql = df_pql[['atom_charge','LJ_sigma','LJ_epsilon']]
        # print(df_pql.head())

        # pdb list & res number
        File_List = sorted(glob.glob('{}/{}/*.pdb'.format(self.source_dir, pdb_dir)))
        RI = sorted(glob.glob('{}/{}/xmu_res_*.dat'.format(self.source_dir, xmu_dir)))
        residue_num = len(RI)                                     # residue 수
        data_num = len(File_List)                                 # 데이터 (conformation) 수
        print('residues : {}, conformations : {}'.format(residue_num, data_num))

        ################################################################################################
        cutoff = 10       # cutoff 설정
        box_size = 21
        max = 20
        min = 0
        movepoint = 10    # center

        subset_num = int(data_num/slice)
        print('subset_num :', subset_num)

        for part_no in range(slice):
            start_idx = part_no * subset_num
            end_idx = start_idx + subset_num
            print(start_idx, end_idx)

            target_file1 = '{}/3D/{}{}_3D_atom_charge.npy'.format(self.work_dir, self.pdb_code, part_no+1)
            target_file2 = '{}/3D/{}{}_3D_LJ_sigma.npy'.format(self.work_dir, self.pdb_code, part_no+1)
            target_file3 = '{}/3D/{}{}_3D_LJ_epsilon.npy'.format(self.work_dir, self.pdb_code, part_no+1)
            if os.path.exists(target_file3):
                continue

            total_box_data1 = []
            total_box_data2 = []
            total_box_data3 = []

            for i in range(start_idx, end_idx, 1):                   # 데이터 (conformation) 수
                ppdb.read_pdb(File_List[i])             # 파일 순서대로 가져오기
                df = ppdb.df['ATOM'][['residue_number','residue_name','atom_name','x_coord','y_coord','z_coord']]
                # df = df[df['chain_id'] == 'A']

                ####################################
                # Merge pdb & protein.pql
                if i == 0:
                    print(df.shape)
                    print(df_pql.shape)
                    if not df.shape[0] == df_pql.shape[0]:
                        err_msg = 'pdb : pql not match atom number --- {}'.format(self.pdb_code)
                        with open(self.error_log, "a") as f:
                            print(err_msg, file=f)
                        return err_msg

                df = pd.concat([df, df_pql], axis=1)
                df.to_csv('{}/{}_merge.pdb'.format(self.source_dir, self.pdb_code), index=None, sep='\t')

                if i == 0:
                    # print(df)
                    print(df.shape)
                    if not df.shape[0] == df_pql.shape[0]:
                        err_msg = 'merge : pql not match atom number --- {}'.format(self.pdb_code)
                        with open(self.error_log, "a") as f:
                            print(err_msg, file=f)
                        return err_msg

                ####################################
                box_per_residue1 = []  # 1개의 conformation에 해당하는 박스를 담을 변수
                box_per_residue2 = []
                box_per_residue3 = []

                for j in range(residue_num):                                                                                  # residue 개수 만큼 반복
                    Temp_df = df.loc[(df["residue_number"]==j+1), ['residue_name','atom_name','x_coord','y_coord','z_coord']]   # Temp_df에 residue별로 담기
                    AT_coord = np.array(df.loc[(df["residue_number"]==j+1), ['residue_name','atom_name','x_coord','y_coord','z_coord','atom_charge','LJ_sigma','LJ_epsilon']])
                    AT_number = len(AT_coord)      # 해당 residue의 atom개수 ATN

                    ############################################################## 좌표 CB 추출 ############################################################
                    Residue_n = np.array(Temp_df.residue_name)[0] # 해당 residue의 이름

                    if Residue_n == 'GLY':
                        CB_coord = np.array(Temp_df.loc[(Temp_df["atom_name"]=='CA'), ['x_coord','y_coord','z_coord']])   # GLY 인경우는 CA 좌표로 대체
                    else:
                        CB_coord = np.array(Temp_df.loc[(Temp_df["atom_name"]=='CB'), ['x_coord','y_coord','z_coord']])   # 베타 카본의 좌표를 구해온다

                    ######################################### CB 근처에 있는 atom의 데이터를 모두 계산하여 받아온다 ###########################################
                    other_residue = self.cal_distance(j, CB_coord, np.array(df), cutoff)
                    other_number = len(other_residue) # 받아온 atom의 수 OTN

                    ###################################################### CA,N,C 좌표 초기화 (CA가 원점) ###################################################
                    a = np.array(Temp_df.loc[(Temp_df["atom_name"]=='CA'), ['x_coord','y_coord','z_coord']]).reshape(-1) # CA 좌표
                    b = np.array(Temp_df.loc[(Temp_df["atom_name"]=='N'), ['x_coord','y_coord','z_coord']]).reshape(-1) # N 좌표
                    c = np.array(Temp_df.loc[(Temp_df["atom_name"]=='C'), ['x_coord','y_coord','z_coord']]).reshape(-1) # C 좌표
                    S_point = a
                    a = a-S_point                 # CA좌표가 (0,0,0)이 되도록 이동
                    b = b-S_point                 # N 좌표도 이동
                    c = c-S_point                 # C 좌표도 이동
                    T_CB = CB_coord.reshape(-1)
                    T_CB = T_CB-S_point

                    #################################################### CA가 원점이 되도록 모두 이동 ########################################################
                    AT_coord,other_residue = self.coord_move(S_point,AT_coord,other_residue,AT_number,other_number)

                    ################################################# CA, N, C 가 평면이 되도록 회전 #########################################################
                    a,b,c,AT_coord,other_residue,T_CB = self.cal_rotating(a,b,c,1,AT_coord,other_residue,T_CB,AT_number,other_number) # N 좌표를 YZ 평면으로 Z축 회전
                    a,b,c,AT_coord,other_residue,T_CB = self.cal_rotating(a,b,c,2,AT_coord,other_residue,T_CB,AT_number,other_number) # N 좌표를 XY 평면으로 X축 회전
                    a,b,c,AT_coord,other_residue,T_CB = self.cal_rotating(a,b,c,3,AT_coord,other_residue,T_CB,AT_number,other_number) # C 좌표를 XY 평면으로 Y축 회전

                    #################################################### CB가 원점이 되도록 모두 이동 #########################################################
                    AT_coord,other_residue = self.coord_move(T_CB,AT_coord,other_residue,AT_number,other_number)

                    ########################################## 박스 처리 ######################################################
                    other_residue = other_residue[:,1:]

                    Temp_box1 = np.zeros((box_size,box_size,box_size))  # atom_charge
                    Temp_box2 = np.zeros((box_size,box_size,box_size))  # LJ_sigma
                    Temp_box3 = np.zeros((box_size,box_size,box_size))  # LJ_epsilon

                    #### 좌표 반올림 및 중심 (10,10,10) 으로 모든 좌표 이동 ####
                    for pre_i in range(AT_number):
                        AT_coord[pre_i,2]=np.round(AT_coord[pre_i,2]) + movepoint
                        AT_coord[pre_i,3]=np.round(AT_coord[pre_i,3]) + movepoint
                        AT_coord[pre_i,4]=np.round(AT_coord[pre_i,4]) + movepoint
                    for pre_i2 in range(other_number):
                        other_residue[pre_i2,2]=np.round(other_residue[pre_i2,2]) + movepoint
                        other_residue[pre_i2,3]=np.round(other_residue[pre_i2,3]) + movepoint
                        other_residue[pre_i2,4]=np.round(other_residue[pre_i2,4]) + movepoint

                    #### 원자의 반지름 크기를 좌표 위치에 입력 ####
                    AT_cnt=0
                    OT_cnt=0
                    for box_i in range(AT_number):
                        x = int(AT_coord[box_i,2].astype(np.int64))
                        y = int(AT_coord[box_i,3].astype(np.int64))
                        z = int(AT_coord[box_i,4].astype(np.int64))
                        if (x<=max and x>=min) and (y<=max and y>=min) and (z<=max and z>=min) :   # 박스를 벗어나는 좌표는 버린다 (사이즈 변형 대비)
                            # print(AT_coord[box_i,1], AT_coord[box_i,5])
                            if AT_coord[box_i,1].find("C")==0:
                                AT_cnt+=1
                                Temp_box1[x,y,z] = AT_coord[box_i,5]  # atom_charge
                                Temp_box2[x,y,z] = AT_coord[box_i,6]  # LJ_sigma
                                Temp_box3[x,y,z] = AT_coord[box_i,7]  # LJ_epsilon
                            elif AT_coord[box_i,1].find("N")==0:
                                AT_cnt+=1
                                Temp_box1[x,y,z] = AT_coord[box_i,5]
                                Temp_box2[x,y,z] = AT_coord[box_i,6]
                                Temp_box3[x,y,z] = AT_coord[box_i,7]
                            elif AT_coord[box_i,1].find("O")==0:
                                AT_cnt+=1
                                Temp_box1[x,y,z] = AT_coord[box_i,5]
                                Temp_box2[x,y,z] = AT_coord[box_i,6]
                                Temp_box3[x,y,z] = AT_coord[box_i,7]
                            elif AT_coord[box_i,1].find("S")==0:
                                AT_cnt+=1
                                Temp_box1[x,y,z] = AT_coord[box_i,5]
                                Temp_box2[x,y,z] = AT_coord[box_i,6]
                                Temp_box3[x,y,z] = AT_coord[box_i,7]
                            elif AT_coord[box_i,1].find("H")==0 or AT_coord[box_i,1].find("1H")==0 or AT_coord[box_i,1].find("2H")==0 or AT_coord[box_i,1].find("3H")==0:
                                AT_cnt+=1
                                Temp_box1[x,y,z] = AT_coord[box_i,5]
                                Temp_box2[x,y,z] = AT_coord[box_i,6]
                                Temp_box3[x,y,z] = AT_coord[box_i,7]

                    for box_i2 in range(other_number):
                        x = int(other_residue[box_i2,2].astype(np.int64))
                        y = int(other_residue[box_i2,3].astype(np.int64))
                        z = int(other_residue[box_i2,4].astype(np.int64))
                        if (x<=max and x>=min) and (y<=max and y>=min) and (z<=max and z>=min) :
                            # print(other_residue[box_i2,1], other_residue[box_i2,6])
                            if other_residue[box_i2,1].find("C")==0:
                                OT_cnt+=1
                                Temp_box1[x,y,z] = other_residue[box_i2,5]  # atom_charge
                                Temp_box2[x,y,z] = other_residue[box_i2,6]  # LJ_sigma
                                Temp_box3[x,y,z] = other_residue[box_i2,7]  # LJ_epsilon
                            elif other_residue[box_i2,1].find("N")==0:
                                OT_cnt+=1
                                Temp_box1[x,y,z] = other_residue[box_i2,5]
                                Temp_box2[x,y,z] = other_residue[box_i2,6]
                                Temp_box3[x,y,z] = other_residue[box_i2,7]
                            elif other_residue[box_i2,1].find("O")==0:
                                OT_cnt+=1
                                Temp_box1[x,y,z] = other_residue[box_i2,5]
                                Temp_box2[x,y,z] = other_residue[box_i2,6]
                                Temp_box3[x,y,z] = other_residue[box_i2,7]
                            elif other_residue[box_i2,1].find("S")==0:
                                OT_cnt+=1
                                Temp_box1[x,y,z] = other_residue[box_i2,5]
                                Temp_box2[x,y,z] = other_residue[box_i2,6]
                                Temp_box3[x,y,z] = other_residue[box_i2,7]
                            elif other_residue[box_i2,1].find("H")==0 or other_residue[box_i2,1].find("1H")==0 or other_residue[box_i2,1].find("2H")==0 or other_residue[box_i2,1].find("3H")==0:
                                OT_cnt+=1
                                Temp_box1[x,y,z] = other_residue[box_i2,5]
                                Temp_box2[x,y,z] = other_residue[box_i2,6]
                                Temp_box3[x,y,z] = other_residue[box_i2,7]

                    #print("AT cnt: ",AT_cnt)
                    #print("OT cnt: ",OT_cnt)
                    #print("residue atoms: ",AT_number)
                    #print("other atoms: ",other_number)
                    #############################################################################################################
                    box_per_residue1.append(Temp_box1)   #1개의 conformation에 해당하는 box를 담는다  (residue_number,box_size,box_size,box_size)
                    box_per_residue2.append(Temp_box2)
                    box_per_residue3.append(Temp_box3)

                total_box_data1.append(box_per_residue1)
                total_box_data2.append(box_per_residue2)
                total_box_data3.append(box_per_residue3)

                if (i+1)%10 == 0:
                    print('{:5d} - {}'.format(i+1, os.path.basename(File_List[i])))

            total_box_data1 = np.array(total_box_data1, dtype=np.float16)
            total_box_data2 = np.array(total_box_data2, dtype=np.float16)
            total_box_data3 = np.array(total_box_data3, dtype=np.float16)
            # print(total_box_data.shape)

            np.save(target_file1, total_box_data1)
            np.save(target_file2, total_box_data2)
            np.save(target_file3, total_box_data3)

        return data_num
