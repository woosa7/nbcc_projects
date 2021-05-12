import os
import subprocess

# ---------------------------------------------------------------------------
# calc xmu
# ---------------------------------------------------------------------------
class RISM3D:
    def __init__(self, config):
        self.config = config


    def calc_xmu(self, traj_dir):
        xmu_dir = os.path.join(traj_dir, '6_calc_xmu') # traj_01/6_calc_xmu/

        data_dir = os.path.join(xmu_dir, 'xmu_data')
        time_dir = os.path.join(xmu_dir, 'xmu_vs_time')
        exec_dir = os.path.join(xmu_dir, 'execute')

        result_file = '{}/xmu_vs_time.dat'.format(time_dir)
        if os.path.exists(result_file):
            return result_file

        try:
            os.makedirs(data_dir)
            os.makedirs(time_dir)
            os.makedirs(exec_dir)
        except OSError:
            pass

        cwd = os.getcwd()

        # --------------------------------------
        # run xmu_calc
        os.chdir(xmu_dir)
        
        subprocess.call('cp {}/4_extract_pdb/list_files {}/'.format(traj_dir, xmu_dir), shell=True)

        op_file = '/homes/eta/users/aiteam/delta_f_implicit/auto_sims/autoMD/run_xmu_calc_with_GPU.sh'
        subprocess.call('cp {} {}/'.format(op_file, xmu_dir), shell=True)

        GPU_NUM = self.config['gpu_no']
        subprocess.call('./run_xmu_calc_with_GPU.sh %s' % GPU_NUM, shell=True)

        os.chdir(cwd)

        # --------------------------------------
        # run xmu_vs_time
        os.chdir(time_dir)

        time_file = '/homes/eta/users/aiteam/delta_f_implicit/auto_sims/autoMD/list_times'
        subprocess.call('cp {} {}/'.format(time_file, time_dir), shell=True)

        subprocess.call('ls ../xmu_data/*.xmu >  list_files', shell=True)
        subprocess.call('/homes/epsilon/users/nbcc/HP36_tutorial/tools/analytic_xmu_vs_time/run.exe', shell=True)

        os.chdir(cwd)

        return result_file

# ---------------------------------------------------------------------------
