import os
import sys
import glob
import urllib
import ssl
import shutil
import subprocess
import pandas as pd
import md_preprocess
import md_prep_openmm
import md_minimization
import md_equilibration
import md_production
import md_calc_sfe
from simtk import openmm as mm, unit
import time

# ---------------------------------------------------------------------
def job_fail_marking(job_list, chain_code, node_no, gpu_no):
    # 작업 실패 표시
    df = pd.read_csv(os.path.abspath(job_list))
    df.loc[(df['p_code'] == chain_code) & (df['job_node'] == node_no) & (df['gpu'] == gpu_no), 'status'] = -1
    df.to_csv(job_list, index=False)


# ---------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    params_file = ''
    try:
        params_file = sys.argv[1]
    except IndexError as e:
        params_file = 'md_parameters'

    if os.path.exists(params_file):
        for line in open(params_file, 'r'):
            if len(line.strip()) > 0:
                exec(line)
    else:
        print('Not exist the md_parameters file')
        sys.exit()

    if force_field == 'ff99SBildn':
        force_field = 'amber99sbildn.xml'
    else:
        force_field = 'amber14/protein.ff14SB.xml'

    if gpu_no == 1:
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="0"

    work_dir = os.path.abspath(work_dir)       # node / auto_simulation

    # data_dir = os.path.abspath(data_dir)       # server
    # if not os.path.exists(data_dir):
    #     os.makedirs(data_dir)

    progress_log = '{}/progress_log'.format(work_dir)
    error_log = '{}/progress_error'.format(work_dir)
    if not os.path.exists(progress_log):
        with open(progress_log, "w") as f:
            print('NBCC AutoMD progress log', file=f)
            print('------------------------', file=f)
        with open(error_log, "w") as f:
            print('NBCC AutoMD progress error log', file=f)
            print('------------------------------', file=f)


    # ---------------------------------------------------------------------
    # Take a job from job list

    # 작업할 pdb list 읽기
    new_job = False

    df = pd.read_csv(os.path.abspath(job_list))
    # node에서 job을 다시 시작하는 경우 미완료된 자신의 job을 확인
    pdb_list = df[(df['job_node'] == node_no) & (df['gpu'] == gpu_no) & (df['status'] == 0)]['p_code']
    if len(pdb_list) == 0 :
        # 새로운 job을 얻음
        pdb_list = df[(df['job_node'] == 0) & (df['status'] == 0)]['p_code']
        new_job = True
        if len(pdb_list) == 0 : sys.exit()   # 작업할 내역 없음.

    p_code = pdb_list.iloc[0]

    if new_job:
        # 작업을 가져온 pdb code에 노드 번호 표시
        df.loc[df['p_code'] == p_code, 'job_node'] = node_no
        df.loc[df['p_code'] == p_code, 'gpu'] = gpu_no
        df.to_csv(job_list, index=False)

    print('-------------------------------')
    print('Start --- {} : node {} : gpu {}'.format(p_code, node_no, gpu_no))
    print('-------------------------------')
    with open(progress_log, "a") as f:
        print('Start --- {} : node {} : gpu {}'.format(p_code, node_no, gpu_no), file=f)

    # ---------------------------------------------------------------------
    # prepare pdb

    # PREFIX, RCSB_CODE, CHAIN
    pdb_prefix = p_code[:2]
    pdb_code = p_code[:4]
    chain_code = pdb_code
    if len(p_code) == 5:
        pdb_chain = p_code[4]
        chain_code = p_code
    else:
        pdb_chain = ''

    # 작업 디렉토리 및 실행파일 설정
    cwd = os.getcwd()

    os.chdir(work_dir)
    if not os.path.exists(work_dir):
        raise AttributeError('Work directory not found')

    print('temperature :', target_temperature)

    # pdb_work_dir = os.path.join(work_dir, '%s/%s_%sK' % (pdb_prefix, chain_code, target_temperature))  # 1A/1A1XA_310K
    pdb_work_dir = os.path.join(work_dir, '%s/%s' % (pdb_prefix, chain_code))  # 1A/1A1XA
    print('work_dir    :',pdb_work_dir)
    if not os.path.exists(pdb_work_dir):
        os.makedirs(pdb_work_dir)

    # copy parameter file
    # shutil.copy2(os.path.join(work_dir, 'nbcc_autoMD/{}'.format(params_file)), os.path.join(pdb_work_dir, params_file))

    pdb_file = '{}/{}.pdb'.format(pdb_work_dir, pdb_code)   # 1A/1A1XA/1A1X.pdb
    print('pdb_file    :', pdb_file)
    if not os.path.exists(pdb_file):
        # download from rcsb
        print('------------ download : ', pdb_code)
        ssl._create_default_https_context = ssl._create_unverified_context
        url = "https://files.rcsb.org/download/{}.pdb".format(pdb_code)

        try:
            pfile = urllib.request.urlopen(url)
            with open(pdb_file, 'wb') as output:
                output.write(pfile.read())
                # print(pdb_code, '--- Download Done!')
        except urllib.error.HTTPError:
            with open(progress_log, "a") as f:
                print('{} --- NotFound : HTTPError'.format(pdb_code), file=f)

    # ---------------------------------------------------------------------
    # pdb preprocess
    model_file = '{}/{}_model.pdb'.format(pdb_work_dir, chain_code)
    print('model_file  :', model_file)

    if not os.path.exists(model_file):
        pdb_preprocessor = md_preprocess.pdb_preprocess(pdb_work_dir)
        result = pdb_preprocessor.run_pre_process(pdb_file, pdb_code, pdb_chain, force_field)
        print('preprocess  :', result)

    # ---------------------------------------------------------------------
    # prep openmm input files
    model_file = '{}/prep/model.xml'.format(pdb_work_dir)
    print('model.xml   :', model_file)

    if not os.path.exists(model_file):
        open_mm = md_prep_openmm.prep_openmm(pdb_work_dir, error_log)
        result = open_mm.generate_openmm_files(chain_code, force_field, solvent_model, buffer_size)
        if result == 'Exception':
            # 작업 실패 표시
            job_fail_marking(job_list, chain_code, node_no, gpu_no)

        print('prep openmm :', result)

    # ---------------------------------------------------------------------
    # minimization
    min_file = '{}/min/min2.pdb'.format(pdb_work_dir)

    if not os.path.exists(min_file):
        md_min = md_minimization.minimization(pdb_work_dir, chain_code, error_log)
        result = md_min.run_minimization(target_temperature, maxcyc_min_1, maxcyc_min_2, buffer_size)
        if result == 'Exception':
            # 작업 실패 표시
            job_fail_marking(job_list, chain_code, node_no, gpu_no)

        print('min_file    :', result)

    # ---------------------------------------------------------------------
    # equilibration
    eq_file = '{}/eq/eq2.pdb'.format(pdb_work_dir)

    if not os.path.exists(eq_file):
        md_eq = md_equilibration.equilibration(pdb_work_dir, chain_code, error_log)
        result = md_eq.run_equilibration(nstlim_eq_1, nstlim_eq_2, init_temperature, target_temperature, ntb_eq, buffer_size)
        if result == 'Exception':
            # 작업 실패 표시
            job_fail_marking(job_list, chain_code, node_no, gpu_no)

        print('eq_file     :', result)

    # ---------------------------------------------------------------------
    # production
    md_prod = md_production.production(pdb_work_dir, chain_code, error_log)
    result = md_prod.run_production(nstlim_md, target_temperature, ntb_md, pressure_md, ntwx_md, simulation_time_ns)
    if result == 'Exception':
        # 작업 실패 표시
        job_fail_marking(job_list, chain_code, node_no, gpu_no)

    print('md_file     :', result)

    # ---------------------------------------------------------------------
    # Gsolv Calculation
    md_calc = md_calc_sfe.rism3d(pdb_work_dir, progress_log)
    result = md_calc.run_rism3d(chain_code, simulation_time_ns, calc_mode, box_size, grid_size, decomp_interval, gpu_no)

    xmu_count = len(glob.glob('{}/xmu/xmu*.dat'.format(pdb_work_dir)))
    if int(xmu_count) < 20:
        # 작업 실패 표시
        job_fail_marking(job_list, chain_code, node_no, gpu_no)

    print('calc_sfe    :', result)

    # ---------------------------------------------------------------------
    # Summary Gsolv values
    total_res = md_calc.run_xmu_summary(chain_code)
    print('xmu_summary :', total_res)

    # ---------------------------------------------------------------------
    # Closing Job
    res_xmu_list = sorted(glob.glob('{}/xmu/xmu_res_*.dat'.format(pdb_work_dir)))
    if total_res == len(res_xmu_list):
        # 작업 완료 표시
        df = pd.read_csv(os.path.abspath(job_list))
        df.loc[(df['p_code'] == chain_code) & (df['job_node'] == node_no) & (df['gpu'] == gpu_no), 'status'] = 1
        df.to_csv(job_list, index=False)
    else:
        print('total_res : {} --- res_xmu : {}'.format(total_res, len(res_xmu_list)))

    print('')
    os.chdir(cwd)

    elapsed_time = time.time() - start_time
    print('===== elapsed_time : {:.2f} min'.format(elapsed_time/60))
    print()
