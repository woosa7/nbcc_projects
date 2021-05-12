#!/home/nbcc/anaconda3/envs/prowave/bin/python
#SBATCH --nodes=1
#SBATCH --time=168:00:00
#SBATCH --job-name=WMD_ANAL
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
"""
Run analysis module

"""
import os
import sys
import argparse
import subprocess
import numpy as np
import mdtraj as mdt


def load_trajectories(trajin):
    """
    load trajectory from list of trajectory input files
    working directory need to be changed before call this function

    :param trajin: list of trajectory input files
    :return: (trajectory, reference)
    """
    topology_file = '../../prep/model_solv.pdb.gz'

    trajectory = mdt.load('../../md/%s/md.dcd' % trajin[0], top=topology_file)
    reference = mdt.load(topology_file, topology_file)
    for traj in trajin[1:]:
        t0 = mdt.load('../../md/%s/md.dcd' % traj, top=topology_file)
        trajectory = trajectory.join(t0)
    return trajectory, reference


def select_mask(mask, t):
    """
    extract selection from trajectory with mask text

    :param mask: Mask text
    :param t: trajectory
    :return:
    """
    try:
        sel = t.topology.select(mask.replace('"', ''))
    except:
        print("ERROR: Failed to select with '{}'".format(mask))
        return
    return sel


def get_masses(t):
    """
    get mass of each atom

    :param t: trajectory
    :return:
    """
    masses = []
    for a in t.topology.atoms:
        masses.append(a.element.mass)
    return np.array(masses)


def get_traj_info(work_dir, md_serial):
    cwd = os.getcwd()
    os.chdir(work_dir)

    dcd_path = os.path.join(work_dir, 'md/%s/md.dcd' % md_serial)
    t = mdt.load(dcd_path, top=os.path.join(work_dir, 'prep/model_solv.pdb.gz'))

    print({
        'frames': t.n_frames,
        'atoms': t.n_atoms,
        'residues': t.n_residues,
    })
    os.chdir(cwd)


def process_rmsd(work_dir, anal_serial, trajin, mask):
    base_dir = os.path.join(work_dir, 'analyses/%s' % anal_serial)
    cwd = os.getcwd()
    os.chdir(base_dir)

    r = mdt.load(os.path.join(work_dir, 'prep/model_solv.pdb.gz'))
    r.image_molecules(inplace=True)

    rmsds = []
    for traj in trajin:
        traj_file_name = os.path.join(work_dir, 'md/%s/md.dcd' % traj)
        t = mdt.load(traj_file_name, top=os.path.join(work_dir, 'prep/model_solv.pdb.gz'))
        t.image_molecules(inplace=True)
        t.center_coordinates(mass_weighted=False)
        tsel = select_mask(mask, t)
        t.superpose(r, frame=0, atom_indices=tsel)
        rmsds.append(mdt.rmsd(t, r, 0, atom_indices=tsel, precentered=True) * 10.0)

    rmsd = np.concatenate(rmsds)
    np.savetxt('rmsd.out', rmsd, fmt='%.4f', header=mask)
    os.chdir(cwd)


def process_rmsf(work_dir, anal_serial, trajin):
    base_dir = os.path.join(work_dir, 'analyses/%s' % anal_serial)
    cwd = os.getcwd()
    os.chdir(base_dir)
    #
    # r = mdt.load(os.path.join(work_dir, 'prep/model_solv.pdb.gz'))
    #
    # xyzs = []
    # for traj in trajin:
    #     traj_file_name = os.path.join(work_dir, 'md/%s/md.dcd' % traj)
    #     t = mdt.load(traj_file_name, top=os.path.join(work_dir, 'prep/model_solv.pdb.gz'))
    #     t.image_molecules(inplace=True)
    #     t.center_coordinates(mass_weighted=False)
    #     tsel = t.topology.select('protein and name CA')
    #     xyzs.append(t.xyz[:, tsel, :])

    t, r = load_trajectories(trajin)
    # center
    t.center_coordinates(mass_weighted=False)
    # image
    t.image_molecules(inplace=True)
    sel = select_mask('protein and name CA', t)
    # superpose
    t.superpose(r, frame=0, atom_indices=sel)

    refxyz = np.mean(t.xyz[:, sel, :], axis=0)
    rmsf = np.sqrt(3.0 * np.mean((t.xyz[:, sel, :] - refxyz) ** 2, axis=(0, 2))) * 10.0  # nm to Angstrom

    with open('rmsf.out', 'w') as f:
        print('# atom RMSF', file=f)
        for i in range(len(sel)):
            print('{:6d} {:.4f}'.format(sel[i] + 1, rmsf[i]), file=f)
    os.chdir(cwd)


def process_radgyr(work_dir, anal_serial, trajin, mask, mass_weighted=False):
    base_dir = os.path.join(work_dir, 'analyses/%s' % anal_serial)
    cwd = os.getcwd()
    os.chdir(base_dir)
    t, r = load_trajectories(trajin)
    # center
    t.center_coordinates(mass_weighted=False)
    # image
    t.image_molecules(inplace=True)
    sel = select_mask(mask, t)
    # superpose
    t.superpose(r, frame=0, atom_indices=sel)

    num_atoms = sel.size
    if mass_weighted:
        masses = get_masses(t)[sel]
    else:
        masses = np.ones(num_atoms)

    xyz = t.xyz[:, sel, :]
    weights = masses / masses.sum()

    mu = xyz.mean(1)
    centered = (xyz.transpose((1, 0, 2)) - mu).transpose((1, 0, 2))
    squared_dists = (centered ** 2).sum(2)
    rg = (squared_dists * weights).sum(1) ** 0.5 * 10.0  # nm to Angstrom
    np.savetxt('radgyr.out', rg, fmt='%.4f', header=mask)
    os.chdir(cwd)


def process_3drism(work_dir, anal_serial, trajin, stride):
    from simtk import openmm as mm

    base_dir = os.path.join(work_dir, 'analyses/%s' % anal_serial)

    cwd = os.getcwd()
    os.chdir(base_dir)
    trajs, r = load_trajectories(trajin)
    os.chdir(cwd)
    # image
    # trajs.image_molecules(inplace=True)
    # center
    # trajs.center_coordinates(mass_weighted=False)
    sel = select_mask('protein', trajs)
    # superpose
    trajs.superpose(r, frame=0, atom_indices=sel)

    with open(os.path.join(work_dir, 'prep/model.xml'), 'r') as f:
        system = mm.XmlSerializer.deserialize(f.read())

    if not stride:
        stride = 5

    mode = 'm'

    # from .rsm_util import generate_input
    from utils.run_sfe import run_sfe
    for i, t in enumerate(trajs):
        if i % stride != (len(trajs) - 1) % stride:
            continue

        run_sfe(work_dir, mode, system=system, t=t, i=i, anal_serial=anal_serial)

    with open(os.path.join(base_dir, 'gsolv.out'), 'w') as fout:
        fout.write('#\tG_solv\n')
        for i in range(len(trajs)):
            if i % stride != (len(trajs) - 1) % stride:
                continue

            if i == 0:
                result_file = os.path.join(base_dir, 'result.out')
            else:
                result_file = os.path.join(base_dir, 'result_%d.out' % i)

            with open(result_file) as result:
                for line in result:
                    cols = line.split('\t')
                    if cols[0] == 'solvation_free_energy':
                        sfe = float(cols[1])
                        fout.write('%d\t%.4f\n' % (i, sfe))
                        break


def process_sasa(work_dir, anal_serial, trajin):
    base_dir = os.path.join(work_dir, 'analyses/%s' % anal_serial)
    cwd = os.getcwd()
    os.chdir(base_dir)
    t, r = load_trajectories(trajin)
    # center
    t.center_coordinates(mass_weighted=False)
    # image
    t.image_molecules(inplace=True)
    sel = select_mask('protein', t)
    # superpose
    t.superpose(r, frame=0, atom_indices=sel)
    sasa = mdt.shrake_rupley(t)
    np.savetxt('sasa.out', sasa.sum(axis=1), fmt='%.4f', header='protein')
    os.chdir(cwd)


def submit_batch(base_dir, sub_cmd, dependency=None):
    stdout = os.path.join(base_dir, 'stdout')
    stderr = os.path.join(base_dir, 'stderr')

    cmd = [
        '/usr/local/bin/sbatch',
        '--output', stdout,
        '--error', stderr,
    ]

    if dependency:
        cmd += ['--dependency', dependency]

    cmd += sub_cmd

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    submit_msg = proc.stdout.readline().decode()
    if submit_msg.startswith('Submitted batch job'):
        batch_jobid = int(submit_msg.split()[3])

        with open(os.path.join(base_dir, 'status'), 'w') as f:
            f.write('submitted %d' % batch_jobid)

        return batch_jobid
    else:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='func')
    parsers = dict()
    parsers['get_traj_info'] = subparsers.add_parser('get_traj_info')
    parsers['get_traj_info'].add_argument('work_dir')
    parsers['get_traj_info'].add_argument('md_serial')

    parsers['rmsd'] = subparsers.add_parser('rmsd')
    parsers['rmsd'].add_argument('work_dir')
    parsers['rmsd'].add_argument('anal_serial')
    parsers['rmsd'].add_argument('-t', '--trajin', nargs='+', required=True)
    parsers['rmsd'].add_argument('-m', '--mask', required=True)
    parsers['rmsd'].add_argument('--batch', action='store_true')

    parsers['rmsf'] = subparsers.add_parser('rmsf')
    parsers['rmsf'].add_argument('work_dir')
    parsers['rmsf'].add_argument('anal_serial')
    parsers['rmsf'].add_argument('-t', '--trajin', nargs='+', required=True)
    parsers['rmsf'].add_argument('--batch', action='store_true')

    parsers['radgyr'] = subparsers.add_parser('radgyr')
    parsers['radgyr'].add_argument('work_dir')
    parsers['radgyr'].add_argument('anal_serial')
    parsers['radgyr'].add_argument('-t', '--trajin', nargs='+', required=True)
    parsers['radgyr'].add_argument('-m', '--mask', required=True)
    parsers['radgyr'].add_argument('--batch', action='store_true')

    parsers['3drism'] = subparsers.add_parser('3drism')
    parsers['3drism'].add_argument('work_dir')
    parsers['3drism'].add_argument('anal_serial')
    parsers['3drism'].add_argument('-t', '--trajin', nargs='+', required=True)
    parsers['3drism'].add_argument('-s', '--stride')
    parsers['3drism'].add_argument('--batch', action='store_true')

    parsers['sasa'] = subparsers.add_parser('sasa')
    parsers['sasa'].add_argument('work_dir')
    parsers['sasa'].add_argument('anal_serial')
    parsers['sasa'].add_argument('-t', '--trajin', nargs='+', required=True)
    parsers['sasa'].add_argument('--batch', action='store_true')

    args = parser.parse_args()

    if args.func == 'get_traj_info':
        get_traj_info(args.work_dir, args.md_serial)

    elif args.func == 'rmsd':
        if args.batch:
            job_id = submit_batch(os.path.join(args.work_dir, 'analyses/%s' % args.anal_serial), [
                # sys.executable, '-m', 'utils.run_anal',
                os.path.abspath(__file__),
                'rmsd',
                args.work_dir,
                str(args.anal_serial),
                '--trajin'] + args.trajin + [
                '--mask', args.mask,
            ])
            print(job_id)
        else:
            process_rmsd(args.work_dir, args.anal_serial, args.trajin, args.mask)

    elif args.func == 'rmsf':
        if args.batch:
            job_id = submit_batch(os.path.join(args.work_dir, 'analyses/%s' % args.anal_serial), [
                # sys.executable, '-m', 'utils.run_anal',
                os.path.abspath(__file__),
                'rmsf',
                args.work_dir,
                str(args.anal_serial),
                '--trajin'] + args.trajin)
            print(job_id)
        else:
            process_rmsf(args.work_dir, args.anal_serial, args.trajin)

    elif args.func == 'radgyr':
        if args.batch:
            job_id = submit_batch(os.path.join(args.work_dir, 'analyses/%s' % args.anal_serial), [
                # sys.executable, '-m', 'utils.run_anal',
                os.path.abspath(__file__),
                'radgyr',
                args.work_dir,
                str(args.anal_serial),
                '--trajin'] + args.trajin + [
                '--mask', args.mask,
            ])
            print(job_id)
        else:
            process_radgyr(args.work_dir, args.anal_serial, args.trajin, args.mask)

    elif args.func == '3drism':
        if args.batch:
            job_id = submit_batch(os.path.join(args.work_dir, 'analyses/%s' % args.anal_serial), [
                # sys.executable, '-m', 'utils.run_anal',
                os.path.abspath(__file__),
                '3drism',
                args.work_dir,
                str(args.anal_serial),
                '--trajin'] + args.trajin)
            print(job_id)
        else:
            process_3drism(args.work_dir, args.anal_serial, args.trajin, args.stride)

    elif args.func == 'sasa':
        if args.batch:
            job_id = submit_batch(os.path.join(args.work_dir, 'analyses/%s' % args.anal_serial), [
                # sys.executable, '-m', 'utils.run_anal',
                os.path.abspath(__file__),
                'sasa',
                args.work_dir,
                str(args.anal_serial),
                '--trajin'] + args.trajin)
            print(job_id)
        else:
            process_sasa(args.work_dir, args.anal_serial, args.trajin)
