#!/home/nbcc/anaconda3/envs/prowave/bin/python
import os
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
import gzip
import numpy as np
import mdtraj as mdt
from simtk import openmm as mm, unit
from simtk.openmm import app
from simtk.openmm.app.internal.pdbstructure import PdbStructure


PARSER = argparse.ArgumentParser()
PARSER.add_argument('work_dir')
PARSER.add_argument('md_serial')
PARSER.add_argument('--nstlim')
PARSER.add_argument('--ref_temp')
PARSER.add_argument('--ntb')
PARSER.add_argument('--pressure')
PARSER.add_argument('--batch', action='store_true')
PARSER.add_argument('--dependency')
PARSER.add_argument('--md_dir', default='md')
PARSER.add_argument('--ref')


def write_md_input(params):
    work_dir = params['work_dir']
    md_serial = params['md_serial']
    md_dir = os.path.join(work_dir, '%s/%s' % (params['md_dir'], md_serial))
    try:
        os.makedirs(md_dir)
    except OSError:
        pass

    dt = params.get('dt', 0.002)
    nstlim = params['nstlim']
    temp0 = params['ref_temp']
    ntb = params.get('ntb', 1)
    pressure = params.get('pressure', 1.0)

    if not params['ref']:
        if int(md_serial) == 1:
            md_inp = '../../eq/eq2.npz'
        elif int(md_serial) > 1:
            md_inp = '../%d/md.npz' % (int(md_serial) - 1)
        else:
            raise ValueError("md_serial should be a integer number")
    elif params['ref'] == '0':
        md_inp = '../../eq/eq2.npz'
    else:
        md_inp = '../%s/md.npz' % params['ref']

    with open('%s/md.in' % md_dir, 'w') as f:
        json.dump({
            'dt': dt,
            'nstlim': int(nstlim),
            'temp0': float(temp0),
            'ntb': int(ntb),
            'pressure': float(pressure),
            'ntwx': 2500,
            'ntpr': 2500,
            'md_inp': md_inp,
        }, f, indent=2)


def run(md_dir):
    # try:
    #     os.makedirs(md_dir)
    # except OSError:
    #     pass
    pwd = os.getcwd()
    os.chdir(md_dir)
    md_run()
    os.chdir(pwd)


def load_model():
    with open('../../prep/model.xml', 'r') as f:
        system = mm.XmlSerializer.deserialize(f.read())
    model_pdb = PdbStructure(gzip.open('../../prep/model_solv.pdb.gz', 'rt'))
    return system, model_pdb


def md_run():
    system, model_pdb = load_model()
    pdb = app.PDBFile(model_pdb)
    topo = pdb.topology

    # read md.in
    with open('md.in', 'r') as f:
        cntrl =json.load(f)

    rst = np.load(cntrl['md_inp'])
    pos = rst['Positions']

    temp0 = float(cntrl['temp0']) * unit.kelvin
    integrator = mm.LangevinIntegrator(temp0, 1.0 / unit.picoseconds,
                                       float(cntrl['dt']) * unit.picoseconds)
    integrator.setConstraintTolerance(0.00001)
    if int(cntrl['ntb']) == 2:
        system.addForce(mm.MonteCarloBarostat(1 * unit.atmospheres, temp0, 25))
    try:
        platform = mm.Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed'}
        simulation = app.Simulation(topo, system, integrator, platform, properties)
    except:
        try:
            platform = mm.Platform.getPlatformByName('OpenCL')
            print('Run simulation with OpenCL')
            simulation = app.Simulation(topo, system, integrator, platform)
        except:
            platform = mm.Platform.getPlatformByName('CPU')
            print('Run simulation with CPU')
            simulation = app.Simulation(topo, system, integrator, platform)

    simulation.context.setPositions(pos)
    simulation.context.setVelocities(rst['Velocities'])
    simulation.context.setPeriodicBoxVectors(*rst['PeriodicBoxVectors'])
    # tempdir = tempfile.mkdtemp()
    simulation.reporters.append(app.DCDReporter('md.dcd', int(cntrl['ntwx'])))
    nstlim = int(cntrl['nstlim'])
    with open('md.out', 'w') as md_out:
        simulation.reporters.append(app.StateDataReporter(md_out, int(cntrl['ntpr']), step=True,
                                                          time=True, potentialEnergy=True, kineticEnergy=True,
                                                          totalEnergy=True,
                                                          temperature=True, volume=True, density=True, progress=True,
                                                          remainingTime=True, speed=True, totalSteps=nstlim,
                                                          separator=' '))
        simulation.step(nstlim)

    st = simulation.context.getState(getPositions=True, getVelocities=True)

    pos = st.getPositions()
    vel = st.getVelocities()
    bv = st.getPeriodicBoxVectors()
    np.savez('md.npz', Positions=pos, Velocities=vel, PeriodicBoxVectors=bv)

    with open('md.pdb', 'w') as f:
        app.PDBFile.writeFile(topo, pos, f)

    # center and image the pdb file
    t = mdt.load('md.pdb')
    t.image_molecules(inplace=True)
    t.center_coordinates(mass_weighted=True)
    with open('md.pdb', 'w') as f:
        app.PDBFile.writeFile(topo, t.xyz[0] * 10.0, f)

    # shutil.copy2(os.path.join(tempdir, 'md.dcd'), 'md.dcd')
    # shutil.rmtree(tempdir)


def submit_batch(work_dir, md_serial, nstlim, ref_temp, ntb, pressure, md_dir, ref, dependency=None):
    md_dir = os.path.join(work_dir, 'md/%s' % md_serial)
    stdout = os.path.join(md_dir, 'stdout')
    stderr = os.path.join(md_dir, 'stderr')

    cmd = [
        '/usr/local/bin/sbatch',
        '--output', stdout,
        '--error', stderr,
        '--nodes', '1',
        '--time', '168:00:00',
        '--job-name', 'MD',
        '--ntasks', '1',
        '--gres', 'gpu:1',
        '--partition', 'prowave',
    ]

    if dependency:
        cmd += ['--dependency', dependency]

    cmd += [
        # sys.executable, '-m', 'utils.run_md',
        os.path.abspath(__file__),
        work_dir, md_serial,
        '--nstlim', nstlim,
        '--ref_temp', ref_temp,
        '--ntb', ntb,
        '--pressure', pressure,
    ]

    if md_dir:
        cmd += ['--md_dir', md_dir]

    if ref:
        cmd += ['--ref', ref]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    submit_msg = proc.stdout.readline().decode()
    if submit_msg.startswith('Submitted batch job'):
        batch_jobid = int(submit_msg.split()[3])

        with open(os.path.join(md_dir, 'status'), 'w') as f:
            f.write('submitted %d' % batch_jobid)

        return batch_jobid
    else:
        return False


if __name__ == '__main__':
    args = PARSER.parse_args()

    _params = {
        'work_dir': os.path.abspath(args.work_dir),
        'md_serial': args.md_serial,
        'nstlim': args.nstlim if args.nstlim else '500000',
        'ref_temp': args.ref_temp if args.ref_temp else '300',
        'ntb': args.ntb if args.ntb else '2',
        'pressure': args.pressure if args.pressure else '1',
        'md_dir': args.md_dir,
        'ref': args.ref
    }
    write_md_input(_params)

    if args.batch:
        if args.dependency:
            _params['dependency'] = args.dependency
        result = submit_batch(**_params)
        print(result)
    else:

        run(os.path.join(args.work_dir, 'md/%s' % args.md_serial))
