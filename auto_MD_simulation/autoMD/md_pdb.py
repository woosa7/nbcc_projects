import os
import glob
import subprocess

# ---------------------------------------------------------------------------
# extract pdbs from trajectory
# ---------------------------------------------------------------------------
class PDB:
    def __init__(self, pdb_code, res_size, config, work_dir):
        self.pdb_code = pdb_code
        self.res_size = res_size
        self.config = config

        self.work_dir = work_dir  # folded_dir or unfolded_dir


    def extract_pdb(self, traj_dir):
        pdb_dir = os.path.join(traj_dir, '4_extract_pdb') # traj_01/4_extract_pdb/

        list_file = '{}/list_files'.format(pdb_dir)
        if os.path.exists(list_file):
            return list_file

        try:
            os.makedirs(pdb_dir)
        except OSError:
            pass

        cwd = os.getcwd()
        os.chdir(pdb_dir)

        # extract pdb
        subprocess.call('cp {}/1_tleap/{}.top {}/protein.top'.format(self.work_dir, self.pdb_code, pdb_dir), shell=True)
        subprocess.call('cp {}/3_simulation/protein.dcd {}/protein.dcd'.format(traj_dir, pdb_dir), shell=True)

        md_time_ns = self.config['md_time_ns']
        num_frames = 10 * md_time_ns

        _script = 'pdb.in'
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
            # output pdb
            print('trajout protein.pdb pdb multi', file=f)

        subprocess.call('cpptraj -i pdb.in | tee cpptraj.log', shell=True)

        # rename pdb
        list_pdb = sorted(glob.glob('protein.pdb.*'))
        if md_time_ns == 1000:
            # over 100 ns, interval 1 ns, total 900 frames
            start_no = 100000   # 100 ns
            interval = 1000     # 1 ns
            for filename in list_pdb:
                file_no = filename.split('.')[-1]
                new_name = '{}_300K_{:07d}.pdb'.format(self.pdb_code, start_no + int(file_no) * interval)
                subprocess.call('mv {} {}'.format(filename, new_name), shell=True)
        else:
            interval = 100
            for filename in list_pdb:
                file_no = filename.split('.')[-1]
                new_name = '{}_300K_{:07d}.pdb'.format(self.pdb_code, int(file_no) * interval)
                subprocess.call('mv {} {}'.format(filename, new_name), shell=True)

        subprocess.call('ls *.pdb > list_files', shell=True)
        subprocess.call('mkdir pdb', shell=True)
        subprocess.call('mv *.pdb pdb/', shell=True)

        subprocess.call('rm -fr protein.top', shell=True)
        subprocess.call('rm -fr protein.dcd', shell=True)

        os.chdir(cwd)

        return list_file

# ---------------------------------------------------------------------------
