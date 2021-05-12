import os
import sys
import subprocess
import pandas as pd
import md_tleap
import md_pql
import md_simulation
import md_pdb
import md_anal_traj
import md_xmu
import md_eu_xmu

# ---------------------------------------------------------------------
# md_time_ns: 1000 == 1 microsecond
config = {
    'num_traj_folded': 2,
    'num_traj_unfolded': 3,
    'md_time_ns': 1000
}

raw_pdb_dir  = '/homes/eta/users/aiteam/delta_f_implicit/auto_sims/raw_pdb'
prep_pdb_dir = '/homes/eta/users/aiteam/delta_f_implicit/auto_sims/prep_pdb'
job_list_file = '/homes/eta/users/aiteam/delta_f_implicit/auto_sims/job_list'

# ---------------------------------------------------------------------
def run_auto_sim(config):
    # Select pdb to run simulation
    new_job = False

    node_no = config['node_no']
    gpu_no = config['gpu_no']

    df = pd.read_csv(job_list_file)
    pdb_list = df[(df['node'] == node_no) & (df['gpu'] == gpu_no) & (df['done'] == 0)]
    if len(pdb_list) == 0 :
        # Get a new job
        pdb_list = df[(df['node'] == 0) & (df['done'] == 0)]
        new_job = True
        if len(pdb_list) == 0 : sys.exit()   # 작업할 내역 없음.

    pdb_code = pdb_list.iloc[0]['p_code']
    res_size = pdb_list.iloc[0]['r_size']
    print(node_no, gpu_no, pdb_code, res_size)

    if new_job:
        # Record node_no and gpu_no in new job
        df.loc[df['p_code'] == pdb_code, 'node'] = node_no
        df.loc[df['p_code'] == pdb_code, 'gpu'] = gpu_no
        df.to_csv(job_list_file, index=False)

    # ---------------------------------------------------------------------
    work_dir = work_dir = os.path.abspath(pdb_code)
    folded_dir = os.path.join(work_dir, 'folded')
    unfolded_dir = os.path.join(work_dir, 'unfolded')

    if not os.path.exists(work_dir):
        os.makedirs(folded_dir)
        os.makedirs(unfolded_dir)
        subprocess.call('cp {}/{}.pdb {}/'.format(raw_pdb_dir, pdb_code[:4], work_dir), shell=True)

    cwd = os.getcwd()

    log_file = '{}/log_md_progress'.format(work_dir)
    def log_progress(message):
        with open(log_file, "a") as f:
            print(message, file=f)

    # =====================================================================
    # folded state simulation
    # =====================================================================
    os.chdir(folded_dir)

    # 1. tleap
    tleap = md_tleap.tLeap(pdb_code, folded_dir, prep_pdb_dir)
    tleap.tleap_folded()
    log_progress('Done: folded - 1:tleap')

    # 2. generate pql file
    pql = md_pql.AnalPql(pdb_code, folded_dir)
    pql.pql_folded(res_size)
    log_progress('Done: folded - 2:pql')

    num_traj = config['num_traj_folded']
    for i in range(num_traj):
        traj_dir = '{}/traj_{:02d}'.format(folded_dir, i+1)

        # 3. run simulation
        md = md_simulation.MD(pdb_code, res_size, config, folded_dir)
        md.run_simulation(traj_dir)
        log_progress('Done: folded - traj_{:02d} - MD'.format(i+1))

        # 4. extract pdb
        pdb_out = md_pdb.PDB(pdb_code, res_size, config, folded_dir)
        pdb_out.extract_pdb(traj_dir)
        log_progress('Done: folded - traj_{:02d} - extract pdb'.format(i+1))

        # 5. anal cpptraj
        cpptraj = md_anal_traj.AnalTraj(pdb_code, res_size, config, folded_dir)
        cpptraj.run_cpptraj(traj_dir)
        log_progress('Done: folded - traj_{:02d} - anal cpptraj'.format(i+1))

        # 6. calc xmu
        rism3d = md_xmu.RISM3D(config)
        rism3d.calc_xmu(traj_dir)
        log_progress('Done: folded - traj_{:02d} - calc xmu'.format(i+1))

        # 7. calc totalenergy & 8. Eu_plus_xmu
        tenergy = md_eu_xmu.TotalEnergy(pdb_code, res_size, config, folded_dir)
        tenergy.Eu_plus_xmu(traj_dir)
        log_progress('Done: folded - traj_{:02d} - Eu_plus_xmu'.format(i+1))

    os.chdir(cwd)


    # =====================================================================
    # unfolded state simulation
    # =====================================================================
    os.chdir(unfolded_dir)

    # 1. tleap
    tleap = md_tleap.tLeap(pdb_code, unfolded_dir, prep_pdb_dir)
    tleap.tleap_unfolded()
    log_progress('Done: unfolded - 1:tleap')

    # 2. generate pql file
    pql = md_pql.AnalPql(pdb_code, unfolded_dir)
    pql.pql_unfolded(folded_dir)
    log_progress('Done: unfolded - 2:pql')

    num_traj = config['num_traj_unfolded']
    for i in range(num_traj):
        traj_dir = '{}/traj_{:02d}'.format(unfolded_dir, i+1)

        # 3. run simulation
        md = md_simulation.MD(pdb_code, res_size, config, unfolded_dir)
        md.run_simulation(traj_dir)
        log_progress('Done: unfolded - traj_{:02d} - MD'.format(i+1))

        # 4. extract pdb
        pdb_out = md_pdb.PDB(pdb_code, res_size, config, unfolded_dir)
        pdb_out.extract_pdb(traj_dir)
        log_progress('Done: unfolded - traj_{:02d} - extract pdb'.format(i+1))

        # 5. anal cpptraj
        cpptraj = md_anal_traj.AnalTraj(pdb_code, res_size, config, unfolded_dir)
        cpptraj.run_cpptraj(traj_dir)
        log_progress('Done: unfolded - traj_{:02d} - anal cpptraj'.format(i+1))

        # 6. calc xmu
        rism3d = md_xmu.RISM3D(config)
        rism3d.calc_xmu(traj_dir)
        log_progress('Done: unfolded - traj_{:02d} - calc xmu'.format(i+1))

        # 7. calc totalenergy & 8. Eu_plus_xmu
        tenergy = md_eu_xmu.TotalEnergy(pdb_code, res_size, config, unfolded_dir)
        tenergy.Eu_plus_xmu(traj_dir)
        log_progress('Done: unfolded - traj_{:02d} - Eu_plus_xmu'.format(i+1))

    os.chdir(cwd)

    # ---------------------------------------------------------------------
    # Marking work done
    df = pd.read_csv(job_list_file)
    df.loc[(df['p_code'] == pdb_code) & (df['node'] == node_no) & (df['gpu'] == gpu_no), 'done'] = 1
    df.to_csv(job_list_file, index=False)


# ---------------------------------------------------------------------
if __name__ == '__main__':
    node_no = int(sys.argv[1])
    gpu_no = int(sys.argv[2])

    config['node_no'] = node_no
    config['gpu_no'] = gpu_no

    for k in range(100):
        run_auto_sim(config)

# =====================================================================
