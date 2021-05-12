import os
import subprocess

# ---------------------------------------------------------------------------
# anal cpptraj
# ---------------------------------------------------------------------------
class AnalTraj:
    def __init__(self, pdb_code, res_size, config, work_dir):
        self.pdb_code = pdb_code
        self.res_size = res_size
        self.config = config

        self.work_dir = work_dir  # folded_dir or unfolded_dir


    def run_cpptraj(self, traj_dir):
        anal_dir = os.path.join(traj_dir, '5_anal_cpptraj') # traj_01/5_anal_cpptraj/

        dist_file = '{}/E2E_distance.dat'.format(anal_dir)
        if os.path.exists(dist_file):
            return dist_file

        try:
            os.makedirs(anal_dir)
        except OSError:
            pass

        cwd = os.getcwd()
        os.chdir(anal_dir)

        subprocess.call('cp {}/1_tleap/{}.top {}/protein.top'.format(self.work_dir, self.pdb_code, anal_dir), shell=True)
        subprocess.call('cp {}/1_tleap/{}.pdb {}/protein.pdb'.format(self.work_dir, self.pdb_code, anal_dir), shell=True)
        subprocess.call('cp {}/3_simulation/protein.dcd {}/protein.dcd'.format(traj_dir, anal_dir), shell=True)

        md_time_ns = self.config['md_time_ns']
        num_frames = 10 * md_time_ns

        _script = 'anal.in'
        with open(_script, "w") as f:
            # set topology file
            print('parm protein.top [protein_top]', file=f)
            # reference structure (starting structure of this trajectory)
            print('reference protein.pdb parm [protein_top] [FIRST]', file=f)
            # read trajectories
            if md_time_ns == 1000:
                # over 100 ns, interval 1 ns, total 900 frames
                print('trajin protein.dcd 1010 {} 10 parm [protein_top]'.format(num_frames), file=f)
            else:
                # interval 100 ps, all frames
                print('trajin protein.dcd 1 {} 1 parm [protein_top]'.format(num_frames), file=f)
            # centering
            print('strip :WAT', file=f)
            print('strip :Na+', file=f)
            print('strip :Cl-', file=f)
            print('center :1-{} mass origin'.format(self.res_size), file=f)
            print('image origin center familiar', file=f)

            # CA RMSD to the reference structure (excluding the terminal residues)
            print('rmsd ref [FIRST] :3-{}@CA out RMSD_CA_to_FIRST.dat'.format(self.res_size-2), file=f)
            # CA RMSF vs. residue
            print('atomicfluct out RMSF_CA_vs_res.dat @CA byres', file=f)
            # Rg
            print('radgyr out Rg.dat :1-{}'.format(self.res_size), file=f)
            # secondary structure
            print('secstruct :1-{} out SS.dat'.format(self.res_size), file=f)
            # secondary structure (for gnuplot and xmgrace)
            print('secstruct :1-{} out SS.gnu sumout SS.agr'.format(self.res_size), file=f)
            # end to end distance
            print('distance E2E :1@CA :{}@CA out E2E_distance.dat'.format(self.res_size), file=f)

        subprocess.call('cpptraj -i anal.in | tee cpptraj.log', shell=True)

        # hydrophobic_contacts
        pdb_dir = os.path.join(traj_dir, '4_extract_pdb/pdb')
        subprocess.call('ls {}/*.pdb > list_files'.format(pdb_dir), shell=True)
        subprocess.call('/homes/epsilon/users/nbcc/HP36_tutorial/tools/hydrophobic_contacts/run.exe', shell=True)

        # clean files
        subprocess.call('rm -fr protein.top', shell=True)
        subprocess.call('rm -fr protein.pdb', shell=True)
        subprocess.call('rm -fr protein.dcd', shell=True)
        subprocess.call('rm -fr list_files', shell=True)

        os.chdir(cwd)

        return dist_file

# ---------------------------------------------------------------------------
