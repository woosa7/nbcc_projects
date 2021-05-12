import os
import subprocess

# ---------------------------------------------------------------------------
# anal cpptraj
# ---------------------------------------------------------------------------
class TotalEnergy:
    def __init__(self, pdb_code, res_size, config, work_dir):
        self.pdb_code = pdb_code
        self.res_size = res_size
        self.config = config

        self.work_dir = work_dir  # folded_dir or unfolded_dir


    def Eu_plus_xmu(self, traj_dir):
        t_dir = os.path.join(traj_dir, '7_anal_tenergy')
        eu_dir = os.path.join(traj_dir, '8_Eu_plus_xmu')

        time_file = '{}/Eu_plus_xmu_vs_time.dat'.format(eu_dir)
        if os.path.exists(time_file):
            return time_file

        try:
            os.makedirs(t_dir)
            os.makedirs(eu_dir)
        except OSError:
            pass

        # calc total energy
        cwd = os.getcwd()
        os.chdir(t_dir)

        subprocess.call('cp {}/1_tleap/{}.top {}/protein.top'.format(self.work_dir, self.pdb_code, t_dir), shell=True)
        subprocess.call('cp {}/3_simulation/protein.dcd {}/protein.dcd'.format(traj_dir, t_dir), shell=True)

        md_time_ns = self.config['md_time_ns']
        num_frames = 10 * md_time_ns

        _script = 'anal_t.in'
        with open(_script, "w") as f:
            # set topology file
            print('parm protein.top', file=f)
            # read trajectories
            if md_time_ns == 1000:
                # over 100 ns, interval 1 ns, total 900 frames
                print('trajin protein.dcd 1010 {} 10'.format(num_frames), file=f)
            else:
                # interval 100 ps, all frames
                print('trajin protein.dcd 1 {} 1'.format(num_frames), file=f)

            # centering
            print('strip :WAT', file=f)
            print('strip :Na+', file=f)
            print('strip :Cl-', file=f)
            print('center :1-{} mass origin'.format(self.res_size), file=f)
            print('image origin center familiar', file=f)

            # total energy
            print('totalenergy :1-{} out tenergy.dat'.format(self.res_size), file=f)

        subprocess.call('/opt/nbcc/bin/cpptraj-15 -i anal_t.in | tee cpptraj.log', shell=True)

        subprocess.call('rm -fr protein.top', shell=True)
        subprocess.call('rm -fr protein.dcd', shell=True)

        # calc Eu_plus_xmu
        os.chdir(cwd)
        os.chdir(eu_dir)

        subprocess.call('cp ../7_anal_tenergy/tenergy.dat ./energy.dat', shell=True)
        subprocess.call('cp ../6_calc_xmu/xmu_vs_time/xmu_vs_time.dat .', shell=True)
        subprocess.call('cp ../6_calc_xmu/xmu_vs_time/xmu_vs_time_every_250ps.dat .', shell=True)

        subprocess.call('/homes/eta/users/aiteam/delta_f_implicit/tools/calc_Eu_plus_xmu/1000ps_interval/run.exe', shell=True)

        subprocess.call('rm -fr energy.dat', shell=True)

        os.chdir(cwd)

        return time_file

# ---------------------------------------------------------------------------
