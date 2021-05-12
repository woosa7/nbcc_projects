import os
import sys
import json
import shutil
import gzip
import glob
import subprocess
import numpy as np
import pandas as pd
import mdtraj as mdt
from simtk import unit
from simtk import openmm as mm
from biopandas.pdb import PandasPdb

# ---------------------------------------------------------------------
# contant values

class rism3d:
    def __init__(self, workdir, log_file):
        self.work_dir = workdir # pdb_work_dir (1A00C_300K)
        self.calc_dir = os.path.join(workdir, 'calc')
        self.pdb_dir = os.path.join(workdir, 'pdb')
        self.xmu_dir = os.path.join(workdir, 'xmu')

        self.log_file = log_file

        if not os.path.exists(self.calc_dir):
            os.makedirs(self.calc_dir)

        if not os.path.exists(self.pdb_dir):
            os.makedirs(self.pdb_dir)

        if not os.path.exists(self.xmu_dir):
            os.makedirs(self.xmu_dir)


    """
    xmu summary
    """
    def run_xmu_summary(self, chain_code):
        print('')
        print('---------------- run_xmu_summary')

        cwd = os.getcwd()
        os.chdir(self.xmu_dir)

        print(self.xmu_dir)

        result_file = ''
        ppdb = PandasPdb()

        pdb_file = '../pdb/{}_1000.pdb'.format(chain_code)
        if not os.path.exists(pdb_file):
            with open(self.log_file, "a") as f:
                print('{} --- pdb not exist'.format(chain_code), file=f)
            return '{} --- pdb not exist'.format(chain_code)

        # =======================================================================
        # residue별 atom 확인
        ppdb.read_pdb(pdb_file)
        df = ppdb.df['ATOM'][['residue_number','residue_name', 'chain_id']]     # check chain
        df = df[df['chain_id'] == 'A']
        df = df.assign(cnt=1)

        df_model = df.groupby(['residue_number', 'residue_name']).size()
        res_model = df_model.index.get_level_values(1).tolist()
        # print(res_model)

        model_a = df['cnt'].groupby(df['residue_number']).count()
        # print(model_a)

        total_res = len(model_a)
        total_atom = 0
        list_interval = []
        print('total_res  : ', total_res)

        for i in range(total_res):
            total_atom = total_atom + model_a[i+1]
            list_interval.append(total_atom)

        print('total_atom : ', total_atom)
        # print(list_interval)

        with open(self.log_file, "a") as f:
            print('--------- {} : res {} : atoms {}'.format(chain_code, total_res, total_atom), file=f)

        # =======================================================================
        # Gsolv 계산 데이터 확인
        frame_xmu = []
        frame_xmu_LJ = []
        frame_xmu_elec = []
        xmua_list = sorted(glob.glob('../calc/xmua_*.dat'))

        if len(xmua_list) < 20:
            with open(self.log_file, "a") as f:
                print('{} --- calculation not completed'.format(chain_code), file=f)
            return '{} --- calculation not completed'.format(chain_code)

        is_valid_atoms = True
        for k, f_xmu in enumerate(xmua_list):
            print(f_xmu)
            f = open(f_xmu, 'r')
            lines = f.readlines()
            atom_num = int(lines[0])
            # print(atom_num)
            frame_xmu.append(lines[1].split()[0])
            frame_xmu_LJ.append(lines[1].split()[1])
            frame_xmu_elec.append(lines[1].split()[2])
            if k == 0:
                print(frame_xmu, frame_xmu_LJ, frame_xmu_elec)

            if not total_atom == atom_num:
                pstr = '{} : {} - {} : ********** Atom number is not matched...!'.format(chain_code, total_atom, atom_num)
                with open(self.log_file, "a") as f:
                    print(pstr, file=f)
                is_valid_atoms = False
                print(pstr)
                print('')
                break

        if is_valid_atoms == False:
            with open(self.log_file, "a") as f:
                print('{} --- atoms not valid'.format(chain_code), file=f)
            return '{} --- atoms not valid'.format(chain_code)

        # =======================================================================
        # make file for residual decomposed Gsolv - residue 갯수만큼 파일 생성
        # frame별 Gsolv 저장
        xmu_file = 'xmu_vs_time.dat'
        if os.path.exists(xmu_file):
            subprocess.call('rm -fr {}'.format(xmu_file), shell=True)

        for i, f_xmu in zip(range(len(xmua_list)), xmua_list):
            f_frame = f_xmu[-8:].replace('.dat','')
            # print(f_frame, frame_xmu[i])
            with open(xmu_file, "a") as f:
                print('{}  {:20.7f}  {:20.7f}  {:20.7f}'.format(f_frame, np.float(frame_xmu[i]), np.float(frame_xmu_LJ[i]), np.float(frame_xmu_elec[i])), file=f)

        # residue별 Gsolv 저장
        start_point = 0

        xmu_res_file = 'xmu_res_0001_vs_time.dat'
        if os.path.exists(xmu_file):
            subprocess.call('rm -fr xmu_res_*', shell=True)

        for i in range(total_res):
            end_point = list_interval[i]
            # print('-----', chain_code, i+1, start_point, end_point)

            xmu_res_file = 'xmu_res_{:04d}_vs_time.dat'.format(i+1)      # residue별 파일
            for f_file in xmua_list:
                f_frame = f_file[-8:].replace('.dat','')
                f_xmu = '{}/xmua_atm_{}.dat'.format(self.calc_dir, f_frame)               # frame별 파일

                xmu_atom = np.loadtxt(f_xmu, skiprows=2)
                ls_xmu = xmu_atom[:,1]
                ls_xmu_LJ = xmu_atom[:,2]
                ls_xmu_elec = xmu_atom[:,3]

                res_xmu = sum(ls_xmu[start_point:end_point])
                res_xmu_LJ = sum(ls_xmu_LJ[start_point:end_point])
                res_xmu_elec = sum(ls_xmu_elec[start_point:end_point])
                with open(xmu_res_file, "a") as f:
                    print('{}  {:20.7f}  {:20.7f}  {:20.7f}'.format(f_frame, res_xmu, res_xmu_LJ, res_xmu_elec), file=f)

            start_point = end_point
            result_file = xmu_res_file
            # print(i+1, '/', total_res)

        # ----------------------------

        os.chdir(cwd)

        return total_res


    """
    xmu_calc
    """
    # ---------------------------
    # tip3p_combined_300K.xsv : 300, 310, 330, 345, 353, 360, 370
    # ---------------------------
    def run_rism3d(self, chain_code, simulation_time_ns, calc_mode, box_size, grid_size, decomp_interval, gpu_no):
        print('')
        print('---------------- run_calculation')

        cwd = os.getcwd()
        os.chdir(self.calc_dir)
        print(self.calc_dir)

        # ---------------------------------------------------------------
        # copy files
        model_pdb = '{}_model.pdb'.format(chain_code)
        model_xml = 'model.xml'
        topology = 'model_solv.pdb.gz'
        shutil.copy2(os.path.join(self.work_dir, model_pdb), os.path.join(self.calc_dir, model_pdb))
        shutil.copy2(os.path.join(self.work_dir, 'prep/model.xml'), os.path.join(self.calc_dir, model_xml))
        shutil.copy2(os.path.join(self.work_dir, 'prep/model_solv.pdb.gz'), os.path.join(self.calc_dir, topology))

        # ---------------------------------------------------------------
        # take NB force
        with open(model_xml, 'r') as f:
            system = mm.XmlSerializer.deserialize(f.read())

        for i in range(system.getNumForces()):
            nbforce = system.getForce(i)
            if isinstance(nbforce, mm.openmm.NonbondedForce):
                break

        # ---------------------------------------------------------------
        # make input header
        header = """ 20130208C
 {0}
 KH
 {1}
  1.0e-6  10000
  {2:5.1f}   {2:5.1f}   {2:5.1f}
  {3}   {3}   {3}
""".format(calc_mode, '/opt/nbcc/common/3D-RISM/tip3p_combined_300K.xsv', box_size, grid_size)

        md_interval = 5    # 5 ps interval production
        stride = int(decomp_interval / md_interval)
        # print('stride :', stride)

        # ---------------------------------------------------------------
        # Gsolv 계산 완료 여부 확인
        full_xmu_count = int((simulation_time_ns * 1000) / (md_interval * stride))
        print('full_xmu_count :', full_xmu_count)

        xmu_count = len(glob.glob('xmu_*.dat'))
        if calc_mode == 'a':
            xmu_count = len(glob.glob('xmua_atm_*.dat'))
        # print('xmu_count :', xmu_count)

        if full_xmu_count == xmu_count:
            return xmu_count

        ppdb = PandasPdb()

        # ---------------------------------------------------------------
        # 계산 진행
        for i in range(0, simulation_time_ns, 1):
            md_serial = i + 1

            # ---------------------------------------------------------------
            # copy trajectory
            dcd_file = 'md_%d.dcd' % md_serial
            shutil.copy2(os.path.join(self.work_dir, 'md/%s' % dcd_file), os.path.join(self.calc_dir, dcd_file))

            # ---------------------------------------------------------------
            # atom selection
            t = mdt.load(dcd_file, top=topology)
            t.image_molecules(inplace=True)
            t.center_coordinates(mass_weighted=True)

            sel = t.topology.select('protein')

            nframe = t.xyz.shape[0]

            # ---------------------------------------------------------------
            # extract pdb files
            for iframe in [x for x in range(nframe) if (x+1) % stride == 0]:
                structure = t[iframe]
                file_no = (iframe+1) * md_interval + ((md_serial - 1) * 1000)
                target_file = '{}/{}_{:04d}.pdb'.format(self.pdb_dir, chain_code, file_no)
                # print(iframe, file_no)
                if not os.path.exists(target_file):
                    structure.save_pdb(target_file)

                # remove water
                if os.path.exists(target_file):
                    ppdb.read_pdb(target_file)
                    ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['residue_name'] != 'HOH']
                    ppdb.to_pdb(path=target_file,
                                records=None,
                                gz=False,
                                append_newline=True)

            # ---------------------------------------------------------------
            # charge / sigma / epsilon
            natom = len(sel)
            joule_per_mole = unit.joule / unit.mole
            charge = np.empty(natom, dtype=np.double)
            sigma = np.empty(natom, dtype=np.double)
            epsilon = np.empty(natom, dtype=np.double)
            for i in range(natom):
                atmind = int(sel[i])
                p = nbforce.getParticleParameters(atmind)
                charge[i] = p[0].value_in_unit(unit.elementary_charge)
                sigma[i] = p[1].value_in_unit(unit.angstrom)
                epsilon[i] = p[2].value_in_unit(joule_per_mole)

            with open('_3d_rism_debug.out', 'w') as f:
                print('{:6s} {:4s} {:3s} {:>16s} {:>16s} {:>16s}'.format('#NUM', 'ATOM', 'RES', 'CHARGE', 'SIGMA', 'EPSILON'), file=f)
                for i in range(natom):
                    atmind = int(sel[i])
                    atmnum = i + 1
                    atom = t.topology.atom(atmind)
                    print('{:6d} {:4s} {:3s} {:16.7f} {:16.7f} {:16.7f}'.format(atmnum, atom.name, atom.residue.name,
                                                                                charge[i], sigma[i], epsilon[i]), file=f)

            try:
                os.makedirs('logs')
            except OSError:
                pass

            for iframe in [x for x in range(nframe) if (x+1) % stride == 0]:
                coord = t.xyz[iframe] * 10.0  # nm to angstrom

                file_no = (iframe+1) * md_interval + ((md_serial - 1) * 1000)

                target_file = 'xmu_{:04d}.dat'.format(file_no)
                if calc_mode == 'a':
                    target_file = 'xmua_atm_{:04d}.dat'.format(file_no)

                print(target_file)

                if not os.path.exists(target_file):
                    # rism input file
                    rism_input = '_3drism_{:04d}.inp'.format(file_no)
                    with open(rism_input, 'w') as f:
                        print(header, end='', file=f)
                        print('{:5d}'.format(natom), file=f)
                        for i in range(natom):
                            atmind = sel[i]
                            x, y, z = coord[atmind]
                            print(' {:16.5f}{:16.7f}{:16.7f}{:16.7f}{:16.7f}{:16.7f}'.format(
                                charge[i], sigma[i], epsilon[i], x, y, z), file=f)

                    out_f = open('logs/_3drism_%04d_out.log' % file_no, 'w')
                    err_f = open('logs/_3drism_%04d_err.log' % file_no, 'w')
                    subprocess.call(['/opt/nbcc/bin/rism3d-x', rism_input, str(gpu_no)], stdout=out_f, stderr=err_f)

                    out_f.close()
                    err_f.close()

                    f_xmu = '_3drism_%04d.xmu' % file_no
                    # print(f_xmu)
                    if os.path.exists(f_xmu):
                        subprocess.call('mv {} xmu_{:04d}.dat'.format(f_xmu, file_no), shell=True)

                    if calc_mode == 'a':
                        file_name = 'xmua_atm.dat'
                        if os.path.exists(file_name):
                            subprocess.call('mv {} {}_{:04d}.dat'.format(file_name, file_name[:-4], file_no), shell=True)


        subprocess.call('rm -fr *.dcd', shell=True)
        # subprocess.call('rm -fr *.inp', shell=True)
        subprocess.call('rm -fr logs/*', shell=True)
        os.remove(model_pdb)
        os.remove(model_xml)
        os.remove(topology)
        os.remove('_3d_rism_debug.out')

        xmu_count = len(glob.glob('xmu_*.dat'))
        if calc_mode == 'a':
            xmu_count = len(glob.glob('xmua_atm_*.dat'))

        print('xmu_count :', xmu_count)

        os.chdir(cwd)

        return xmu_count

# ----------------------------------------------------------------------------
