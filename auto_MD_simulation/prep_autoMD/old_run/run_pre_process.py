#!/home/nbcc/anaconda3/envs/prowave/bin/python
import os
import argparse
import subprocess
from utils.pdblib.pre_process import main as pre_process

# try:
#     from utils.pdblib.pre_process import main as pre_process
# except ImportError:
#     import sys
#     sys.path.append('/home/nbcc/www/prowave/')
#     from utils.pdblib.pre_process import main as pre_process

PARSER = argparse.ArgumentParser()
PARSER.add_argument('pdb_file')
PARSER.add_argument('--chain')
PARSER.add_argument('--batch', action='store_true')

def run(_pdb_file):

    params = {
        'pdb_file': _pdb_file,
        'auto': True
    }

    if args.chain:
        params['chain'] = args.chain

    print(params)
    pre_process(**params)


def submit_batch(pdb_file, chain=None):
    work_dir = os.path.dirname(os.path.abspath(pdb_file))
    stdout = os.path.join(work_dir, 'stdout')
    stderr = os.path.join(work_dir, 'stderr')

    cmd = [
        '/usr/local/bin/sbatch',
        '--nodes', '100',
        '--time', '168:00:00',
        '--job-name', 'PRE_PROC',
        '--ntasks', '1',
        '--output', stdout,
        '--error', stderr,
        '--partition', 'prowave',
        os.path.abspath(__file__),
        pdb_file,
    ]

    if chain:
        cmd += ['--chain', chain]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    submit_msg = out.decode()
    if err:
        print(err.decode())
    if submit_msg.startswith('Submitted batch job'):
        batch_jobid = int(submit_msg.split()[3])
        return batch_jobid
    else:
        return False

# ===========================================================================
if __name__ == '__main__':
    args = PARSER.parse_args()

    print('------------ run_pre_process.py')
    print(args)

    # if args.batch:
    #     result = submit_batch(args.pdb_file, chain=args.chain)
    #     print(result)
    # else:
    #     pdb_file = os.path.abspath(args.pdb_file)
    #     work_dir = os.path.dirname(pdb_file)
    #     cwd = os.getcwd()
    #     os.chdir(work_dir)
    #     with open(os.path.join(work_dir, 'host'), 'w') as f:
    #         try:
    #             f.write(subprocess.check_output('hostname').decode())
    #             f.write('\n%s' % sys.executable)
    #         except:
    #             pass
    #
    #     try:
    #         run(pdb_file)
    #     except:
    #         import traceback
    #         with open(os.path.join(work_dir, 'stderr'), 'w') as f:
    #             traceback.print_exc(file=f)
    #     finally:
    #         os.chdir(cwd)
