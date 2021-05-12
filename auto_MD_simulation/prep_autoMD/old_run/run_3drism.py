#!/home/nbcc/anaconda3/envs/prowave/bin/python
"""
Run 3D-RISM for ProWaVE
"""
import os
import json
import argparse
import subprocess
import tempfile
import shutil
import numpy as np
import mdtraj as mdt
from simtk import unit
from simtk import openmm as mm


PARSER = argparse.ArgumentParser()
PARSER.add_argument('work_dir')
PARSER.add_argument('mode')
PARSER.add_argument('--batch', action='store_true')
PARSER.add_argument('--dependency')


def write_3drism_input(params):
    """

    :param params:
    :return:
    """
    work_dir = params['work_dir']
    result_dir = os.path.join(work_dir, 'analyses/1')

    try:
        os.makedirs(result_dir)
    except OSError:
        pass

    with open('%s/rism.in' % result_dir, 'w') as f:
        json.dump({
            'mode': params['mode'],
            'box_size': params['box_size'],
            'grid_size': params['grid_size'],
            'force_field': params['force_field'],
        }, f)


def run(work_dir):
    tempdir = tempfile.mkdtemp()
    shutil.copy2(os.path.join(work_dir, 'analyses/1/rism.in'), os.path.join(tempdir, 'rism.in'))
    shutil.copy2(os.path.join(work_dir, 'prep/model.xml'), os.path.join(tempdir, 'model.xml'))
    shutil.copy2(os.path.join(work_dir, 'model.pdb'), os.path.join(tempdir, 'model.pdb'))

    cwd = os.getcwd()
    os.chdir(tempdir)

    with open('rism.in', 'r') as f:
        params = json.load(f)

    mode = params['mode']
    box_size = params['box_size']
    grid_size = params['grid_size']

    # input header
    header = """ 20130208C
 {0}
 KH
 {1}
  1.0e-6  10000
  {2:5.1f}   {2:5.1f}   {2:5.1f}
  {3}   {3}   {3}
""".format(mode, '/opt/nbcc/common/3D-RISM/tip3p_combined_300K.xsv', box_size, grid_size)

    with open('model.xml', 'r') as f:
        system = mm.XmlSerializer.deserialize(f.read())
    # take NB force
    for i in range(system.getNumForces()):
        nbforce = system.getForce(i)
        if isinstance(nbforce, mm.openmm.NonbondedForce):
            break

    # atom selection
    t = mdt.load('model.pdb')
    t.center_coordinates(mass_weighted=False)
    sel = t.topology.select('protein')

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

    with open('_3d_rism_0_debug.out', 'w') as f:
        print('{:6s} {:4s} {:3s} {:>16s} {:>16s} {:>16s}'.format('#NUM', 'ATOM', 'RES', 'CHARGE', 'SIGMA', 'EPSILON'),
              file=f)
        for i in range(natom):
            atmind = int(sel[i])
            atmnum = i + 1
            atom = t.topology.atom(atmind)
            print('{:6d} {:4s} {:3s} {:16.7f} {:16.7f} {:16.7f}'.format(atmnum, atom.name, atom.residue.name,
                                                                        charge[i], sigma[i], epsilon[i]), file=f)

    rism_input = '_3drism_{}.inp'.format(0)

    coord = t.xyz[0] * 10.0  # nm to angstrom
    # avg_coord = np.average(coord.T, axis=1)
    # centering
    # coord -= avg_coord

    # rism input file
    with open(rism_input, 'w') as f:
        print(header, end='', file=f)
        print('{:5d}'.format(natom), file=f)
        for i in range(natom):
            atmind = sel[i]
            x, y, z = coord[atmind]
            print(' {:16.5f}{:16.7f}{:16.7f}{:16.7f}{:16.7f}{:16.7f}'.format(
                charge[i], sigma[i], epsilon[i], x, y, z), file=f)

    # run 3drism
    out_f = open('_3drism_out.log', 'w')
    err_f = open('_3drism_err.log', 'w')
    subprocess.call(['/opt/nbcc/bin/rism3d-x', rism_input, '0'], stdout=out_f, stderr=err_f)
    out_f.close()
    err_f.close()

    # collect output
    rism_output = '_3drism_{}.xmu'.format(0)
    subprocess.call(['/bin/cat', rism_output])

    with open('result.out', 'w') as result:
        with open('_3drism_0.xmu', 'r') as f:
            for line in f:
                key, value = line.split()
                if key == 'solvation_free_energy':
                    print('%s\t%s' % (key, value), file=result)
                    break

        if os.path.exists('_3drism_0.thm'):
            with open('_3drism_0.thm', 'r') as f:
                for line in f:
                    key, value = line.split()
                    if key in ('solvation_energy', 'solvation_entropy'):
                        print('%s\t%s' % (key, value), file=result)
                        continue

        with open('model.pdb', 'r') as f:
            atom_list = [(row[6:11].strip(), row[21:22].strip(), row[22:26].strip()) for row in f if 'ATOM' in row[0:6]]

        xmua_atm = dict()
        if os.path.exists('xmua_atm.dat'):
            with open('xmua_atm.dat', 'r') as f:
                lines = f.readlines()[2:]

            for line in lines:
                key, value = line.split()[:2]
                xmua_atm[key.strip()] = float(value.strip())

        enea_atm = dict()
        if os.path.exists('enea_atm.dat'):
            with open('enea_atm.dat', 'r') as f:
                lines = f.readlines()[2:]

            for line in lines:
                key, value = line.split()[:2]
                enea_atm[key.strip()] = float(value.strip())

        enta_atm = dict()
        if os.path.exists('enta_atm.dat'):
            with open('enta_atm.dat', 'r') as f:
                lines = f.readlines()[2:]

            for line in lines:
                key, value = line.split()[:2]
                enta_atm[key.strip()] = float(value.strip())

        residues = dict()
        if xmua_atm and enea_atm and enta_atm:
            for serial, chain, resnum in atom_list:
                key = '%s %s' % (chain, resnum)
                if key not in residues:
                    residues[key] = [0.0, 0.0, 0.0]
                try:
                    residues[key][0] += xmua_atm[serial]
                    residues[key][1] += enea_atm[serial]
                    residues[key][2] += enta_atm[serial]
                except KeyError:
                    pass

            for key, values in residues.items():
                print('%s\t%s\t%s\t%s' % (key, values[0], values[1], values[2]), file=result)

        elif xmua_atm:
            for serial, chain, resnum in atom_list:
                key = '%s %s' % (chain, resnum)
                if key not in residues:
                    residues[key] = 0.0
                try:
                    residues[key] += xmua_atm[serial]
                except KeyError:
                    pass

            for key, value in residues.items():
                print('%s\t%s' % (key, value), file=result)

    shutil.copy2('_3drism_0.inp', os.path.join(work_dir, 'analyses/1/_3drism_0.inp'))
    shutil.copy2('result.out', os.path.join(work_dir, 'analyses/1/result.out'))

    # for debug
    # from glob import glob
    # for filename in glob('*.dat'):
    #     shutil.copy2(filename, os.path.join(work_dir, 'analyses/1/%s' % filename))

    os.chdir(cwd)
    shutil.rmtree(tempdir)


def submit_batch(work_dir, mode, dependency=None, **kwargs):
    result_dir = os.path.join(work_dir, 'analyses/1')
    stdout = os.path.join(result_dir, 'stdout')
    stderr = os.path.join(result_dir, 'stderr')

    try:
        os.makedirs(result_dir)
    except OSError:
        pass

    cmd = [
        '/usr/local/bin/sbatch',
        '--output', stdout,
        '--error', stderr,
        '--nodes', '1',
        '--time', '168:00:00',
        '--job-name', 'PROWAVE',
        '--ntasks', '1',
        '--gres', 'gpu:1',
    ]

    if dependency:
        cmd += ['--dependency', dependency]

    cmd += [
        os.path.abspath(__file__),
        work_dir,
        mode,
    ]

    # proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    msg = subprocess.check_output(cmd).decode().strip()
    # submit_msg = proc.stdout.readline().decode()
    print(msg)
    # if submit_msg.startswith('Submitted batch job'):
    #     batch_jobid = int(submit_msg.split()[3])
    #     return batch_jobid
    # else:
    #     return False


if __name__ == '__main__':
    args = PARSER.parse_args()

    _params = {
        'work_dir': os.path.abspath(args.work_dir),
        'mode': args.mode,
        'box_size': 128,
        'grid_size': 128,
        'force_field': 'amber99sbildn',
    }

    if args.batch:
        if args.dependency:
            _params['dependency'] = args.dependency
        submit_batch(**_params)
        # print(result)
    else:
        write_3drism_input(_params)
        with open(os.path.join(args.work_dir, 'analyses/1/host'), 'w') as f:
            try:
                f.write(subprocess.check_output('hostname').decode())
            except:
                pass

        run(os.path.abspath(args.work_dir))
