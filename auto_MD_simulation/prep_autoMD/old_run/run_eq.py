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
PARSER.add_argument('--nstlim1')
PARSER.add_argument('--nstlim2')
PARSER.add_argument('--init_temp')
PARSER.add_argument('--ref_temp')
PARSER.add_argument('--batch', action='store_true')
PARSER.add_argument('--dependency')


def write_eq_input(params):
    work_dir = params['work_dir']
    eq_dir = os.path.join(work_dir, 'eq')
    try:
        os.makedirs(eq_dir)
    except OSError:
        pass

    dt = params.get('dt', 0.002)
    nstlim1 = params['nstlim1']
    nstlim2 = params['nstlim2']
    tempi = params['init_temp']
    temp0 = params['ref_temp']
    ntb = params.get('ntb', 1)
    restraint_wt = params.get('restraint_wt', 10.0)

    with open('%s/eq1.in' % eq_dir, 'w') as f:
        json.dump({
            'dt': dt,
            'nstlim': int(nstlim1),
            'tempi': float(tempi),
            'temp0': float(temp0),
            'ntb': int(ntb),
            'ntwx': 2500,
            'ntpr': 2500,
            'restraint_wt': restraint_wt,
        }, f, indent=2)

    with open('%s/eq2.in' % eq_dir, 'w') as f:
        json.dump({
            'dt': dt,
            'nstlim': nstlim2,
            'tempi': tempi,
            'temp0': temp0,
            'ntb': ntb,
            'ntwx': 2500,
            'ntpr': 2500,
        }, f, indent=2)


def run(eq_dir):
    try:
        os.makedirs(eq_dir)
    except OSError:
        pass

    pwd = os.getcwd()
    os.chdir(eq_dir)
    topo, st1 = run_eq1()

    with open('eq2.in', 'r') as f:
        eq2_in = json.load(f)

    if eq2_in['nstlim'] != '0':
        run_eq2(topo, st1)
    else:
        shutil.copy2('eq1.npz', 'eq2.npz')

    os.chdir(pwd)


def load_model():
    try:
        with open('model.xml', 'r') as f:
            system = mm.XmlSerializer.deserialize(f.read())
        model_pdb = PdbStructure(gzip.open('model.pdb.gz', 'rt'))
    except FileNotFoundError:
        with open('../prep/model.xml', 'r') as f:
            system = mm.XmlSerializer.deserialize(f.read())
        model_pdb = PdbStructure(gzip.open('../prep/model_solv.pdb.gz', 'rt'))
    return system, model_pdb


def run_eq1():
    system, model_pdb = load_model()
    pdb = app.PDBFile(model_pdb)

    try:
        rst = np.load('eq1_inp.npz')
    except FileNotFoundError:
        rst = np.load('../min/min2.npz')
    pos = rst['Positions']

    with open('eq1.in', 'r') as f:
        cntrl = json.load(f)

    # add force (restraints)
    restraint_wt = float(cntrl['restraint_wt']) * 418.4  # kcal/mol/A^2 to kJ/mol/nm^2
    force = mm.CustomExternalForce('k*(periodicdistance(x,y,z,x0,y0,z0)^2)')
    force.addGlobalParameter('k', restraint_wt)
    force.addPerParticleParameter('x0')
    force.addPerParticleParameter('y0')
    force.addPerParticleParameter('z0')
    topo = pdb.topology
    for atom in topo.atoms():
        resname = atom.residue.name
        # atmname = atom.name
        if resname not in ('HOH', 'NA', 'CL'):
            ind = atom.index
            x0, y0, z0 = pos[ind]
            force.addParticle(ind, (x0, y0, z0))
    system.addForce(force)

    integrator = mm.LangevinIntegrator(float(cntrl['temp0']) * unit.kelvin, 1.0 / unit.picoseconds,
                                       float(cntrl['dt']) * unit.picoseconds)
    integrator.setConstraintTolerance(0.00001)
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
    simulation.context.setVelocitiesToTemperature(float(cntrl['tempi']) * unit.kelvin)
    simulation.context.setPeriodicBoxVectors(*rst['PeriodicBoxVectors'])
    tempdir = tempfile.mkdtemp()
    simulation.reporters.append(app.DCDReporter(os.path.join(tempdir, 'eq1.dcd'), int(cntrl['ntwx'])))
    nstlim = int(cntrl['nstlim'])
    with open('eq1.out', 'w') as eq1_out:
        simulation.reporters.append(app.StateDataReporter(eq1_out, int(cntrl['ntpr']), step=True,
                                                          time=True, potentialEnergy=True, kineticEnergy=True,
                                                          totalEnergy=True,
                                                          temperature=True, volume=True, density=True,
                                                          progress=True,
                                                          remainingTime=True, speed=True, totalSteps=nstlim,
                                                          separator=' '))
        simulation.step(nstlim)

    st = simulation.context.getState(getPositions=True, getVelocities=True)
    shutil.copy2(os.path.join(tempdir, 'eq1.dcd'), 'eq1.dcd')
    shutil.rmtree(tempdir)

    pos = st.getPositions()
    vel = st.getVelocities()
    bv = st.getPeriodicBoxVectors()
    np.savez('eq1.npz', Positions=pos, Velocities=vel, PeriodicBoxVectors=bv)

    with open('eq1.pdb', 'w') as f:
        app.PDBFile.writeFile(topo, pos, f)

    # center and image the pdb file
    t = mdt.load('eq1.pdb')
    t.image_molecules(inplace=True)
    t.center_coordinates(mass_weighted=True)
    with open('eq1.pdb', 'w') as f:
        app.PDBFile.writeFile(topo, t.xyz[0] * 10.0, f)

    return topo, st


def run_eq2(topo, st1):
    system, model_pdb = load_model()
    pdb = app.PDBFile(model_pdb)

    # read eq2.in
    with open('eq2.in', 'r') as f:
        cntrl = json.load(f)

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

    simulation.context.setPositions(st1.getPositions())
    simulation.context.setVelocities(st1.getVelocities())
    simulation.context.setPeriodicBoxVectors(*st1.getPeriodicBoxVectors())
    tempdir = tempfile.mkdtemp()
    simulation.reporters.append(app.DCDReporter(os.path.join(tempdir, 'eq2.dcd'), int(cntrl['ntwx'])))
    nstlim = int(cntrl['nstlim'])
    with open('eq2.out', 'w') as eq2_out:
        simulation.reporters.append(app.StateDataReporter(eq2_out, int(cntrl['ntpr']), step=True,
                                                          time=True, potentialEnergy=True, kineticEnergy=True,
                                                          totalEnergy=True,
                                                          temperature=True, volume=True, density=True, progress=True,
                                                          remainingTime=True, speed=True, totalSteps=nstlim,
                                                          separator=' '))
        simulation.step(nstlim)

    st = simulation.context.getState(getPositions=True, getVelocities=True)
    shutil.copy2(os.path.join(tempdir, 'eq2.dcd'), 'eq2.dcd')
    shutil.rmtree(tempdir)

    pos = st.getPositions()
    vel = st.getVelocities()
    bv = st.getPeriodicBoxVectors()
    np.savez('eq2.npz', Positions=pos, Velocities=vel, PeriodicBoxVectors=bv)

    with open('eq2.pdb', 'w') as f:
        app.PDBFile.writeFile(topo, pos, f)

    # center and image the pdb file
    t = mdt.load('eq2.pdb')
    t.image_molecules(inplace=True)
    t.center_coordinates(mass_weighted=True)
    with open('eq2.pdb', 'w') as f:
        app.PDBFile.writeFile(topo, t.xyz[0] * 10.0, f)


def submit_batch(work_dir, nstlim1, nstlim2, init_temp, ref_temp, dependency=None):
    eq_dir = os.path.join(work_dir, 'eq')
    stdout = os.path.join(eq_dir, 'stdout')
    stderr = os.path.join(eq_dir, 'stderr')

    cmd = [
        '/usr/local/bin/sbatch',
        '--output', stdout,
        '--error', stderr,
        '--nodes', '1',
        '--time', '168:00:00',
        '--job-name', 'EQ',
        '--ntasks', '1',
        '--gres', 'gpu:1',
        '--partition', 'prowave',
    ]

    if dependency:
        cmd += ['--dependency', dependency]

    cmd += [
        # sys.executable, '-m', 'utils.run_eq',
        os.path.abspath(__file__),
        work_dir,
        '--nstlim1', nstlim1,
        '--nstlim2', nstlim2,
        '--init_temp', init_temp,
        '--ref_temp', ref_temp,
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    submit_msg = proc.stdout.readline().decode()
    if submit_msg.startswith('Submitted batch job'):
        batch_jobid = int(submit_msg.split()[3])

        with open(os.path.join(eq_dir, 'status'), 'w') as f:
            f.write('submitted %d' % batch_jobid)

        return batch_jobid
    else:
        return False


if __name__ == '__main__':
    args = PARSER.parse_args()

    _params = {
        'work_dir': args.work_dir,
        'nstlim1': args.nstlim1 if args.nstlim1 else '10000',
        'nstlim2': args.nstlim2 if args.nstlim2 else '100000',
        'init_temp': args.init_temp if args.init_temp else '0',
        'ref_temp': args.ref_temp if args.ref_temp else '300',
    }
    write_eq_input(_params)

    if args.batch:
        if args.dependency:
            _params['dependency'] = args.dependency
        result = submit_batch(**_params)
        print(result)
    else:
        run(os.path.join(args.work_dir, 'eq'))
