#!/usr/bin/env python
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
PARSER.add_argument('gpu')
PARSER.add_argument('--stride', default=10)


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
            'gpu': params['gpu'],
            'stride': params['stride'],
        }, f)


def run(work_dir):
    tempdir = tempfile.mkdtemp()
    shutil.copy2(os.path.join(work_dir, 'analyses/1/rism.in'), os.path.join(tempdir, 'rism.in'))
    shutil.copy2(os.path.join(work_dir, 'prep/model.xml'), os.path.join(tempdir, 'model.xml'))
    shutil.copy2(os.path.join(work_dir, 'prep/model_solv.pdb.gz'), os.path.join(tempdir, 'model_solv.pdb.gz'))
    shutil.copy2(os.path.join(work_dir, 'md/1/md.dcd'), os.path.join(tempdir, 'md.dcd'))

    cwd = os.getcwd()
    os.chdir(tempdir)

    with open('rism.in', 'r') as f:
        params = json.load(f)

    mode = params['mode']
    box_size = params['box_size']
    grid_size = params['grid_size']
    stride = params['stride']

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
    t = mdt.load('md.dcd', top='model_solv.pdb.gz')
    t.image_molecules(inplace=True)
    t.center_coordinates(mass_weighted=True)

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

    nframe = t.xyz.shape[0]

    try:
        os.makedirs('logs')
        os.makedirs(os.path.join(work_dir, 'analyses/1/logs'))
    except OSError:
        pass

    for iframe in [x for x in range(nframe) if x % stride == (nframe-1) % stride]:
        rism_input = '_3drism_{}.inp'.format(iframe)
        coord = t.xyz[iframe] * 10.0  # nm to angstrom

        # rism input file
        with open(rism_input, 'w') as f:
            print(header, end='', file=f)
            print('{:5d}'.format(natom), file=f)
            for i in range(natom):
                atmind = sel[i]
                x, y, z = coord[atmind]
                print(' {:16.5f}{:16.7f}{:16.7f}{:16.7f}{:16.7f}{:16.7f}'.format(
                    charge[i], sigma[i], epsilon[i], x, y, z), file=f)

        out_f = open('logs/_3drism_%d_out.log' % iframe, 'w')
        err_f = open('logs/_3drism_%d_err.log' % iframe, 'w')
        subprocess.call(['/opt/nbcc/bin/rism3d-x', rism_input, params['gpu']], stdout=out_f, stderr=err_f)
        out_f.close()
        err_f.close()

        for file_name in (
                '_3drism_%d.xmu' % iframe,
                # '_3drism_%d.thm' % iframe,
                'logs/_3drism_%d_out.log' % iframe,
                'logs/_3drism_%d_err.log' % iframe):
            shutil.copy2(file_name, os.path.join(work_dir, 'analyses/1/%s' % file_name))

        for file_name in ('xmua_atm.dat',):  # 'enea_atm.dat', 'enta_atm.dat'):
            shutil.copy2(file_name, os.path.join(work_dir, 'analyses/1/%s_%d.dat' % (file_name[:-4], iframe)))

    os.chdir(cwd)
    shutil.rmtree(tempdir)


if __name__ == '__main__':
    args = PARSER.parse_args()

    _params = {
        'work_dir': args.work_dir,
        'mode': args.mode,
        'box_size': 128,
        'grid_size': 128,
        'force_field': 'amber99sbildn',
        'gpu': args.gpu,
        'stride': int(args.stride),
    }

    write_3drism_input(_params)
    run(os.path.abspath(args.work_dir))
