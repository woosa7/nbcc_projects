import os
import subprocess

# ---------------------------------------------------------------------------
# generate pql file
# ---------------------------------------------------------------------------
class AnalPql:
    def __init__(self, pdb_code, work_dir):
        self.pdb_code = pdb_code
        self.work_dir = work_dir  # folded_dir or unfolded_dir

        self.pql_dir = os.path.join(work_dir, '2_pql')
        try:
            os.makedirs(self.pql_dir)
        except OSError:
            pass


    def pql_folded(self, res_size):
        pql_file = '{}/protein.pql'.format(self.pql_dir)
        if os.path.exists(pql_file):
            return pql_file

        cwd = os.getcwd()
        os.chdir(self.pql_dir)

        subprocess.call('cp {}/1_tleap/{}_water.top {}/protein_water.top'.format(self.work_dir, self.pdb_code, self.pql_dir), shell=True)
        subprocess.call('cp {}/1_tleap/{}_initial.crd {}/protein_initial.crd'.format(self.work_dir, self.pdb_code, self.pql_dir), shell=True)

        _script = 'anal_pql.in'
        with open(_script, "w") as f:
            # set topology file
            print('parm protein_water.top', file=f)
            # set crd file
            print('trajin protein_initial.crd', file=f)
            # centering
            print('center :1-{} mass origin'.format(res_size), file=f)
            print('image origin center familiar', file=f)
            # pql file
            print('pqlout :1-{} out protein.pql noframeout'.format(res_size), file=f)

        subprocess.call('/opt/nbcc/bin/cpptraj-15 -i anal_pql.in', shell=True)

        subprocess.call('rm -fr protein_water.top', shell=True)
        subprocess.call('rm -fr protein_initial.crd', shell=True)

        os.chdir(cwd)

        return pql_file


    def pql_unfolded(self, folded_dir):
        pql_file = '{}/protein.pql'.format(self.pql_dir)

        if os.path.exists(pql_file):
            return pql_file

        subprocess.call('cp {}/2_pql/protein.pql {}/'.format(folded_dir, self.pql_dir), shell=True)

        return pql_file


# ---------------------------------------------------------------------------
