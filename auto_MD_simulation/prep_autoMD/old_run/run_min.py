#!/home/nbcc/anaconda3/envs/prowave/bin/python
import os
import sys
import json
import argparse
import subprocess
import gzip
import numpy as np
import mdtraj as mdt
from simtk import openmm as mm, unit
from simtk.openmm import app
from simtk.openmm.app.internal.pdbstructure import PdbStructure


PARSER = argparse.ArgumentParser()
PARSER.add_argument('work_dir')
PARSER.add_argument('--maxcyc1')
PARSER.add_argument('--maxcyc2')
PARSER.add_argument('--batch', action='store_true')
PARSER.add_argument('--dependency')


def write_min_input(params):
    work_dir = params['work_dir']
    maxcyc1 = params['maxcyc1']
    maxcyc2 = params['maxcyc2']
    # cut = params['cut']
    # nres = params['nres']

    min_dir = os.path.join(work_dir, 'min')

    try:
        os.makedirs(min_dir)
    except OSError:
        pass

    # success = False
    # min1.in
    with open('%s/min1.in' % min_dir, 'w') as f:
        json.dump({
            'restraint_wt': 10.0,
            'maxcyc': int(maxcyc1),
        }, f, indent=2)

    with open('%s/min2.in' % min_dir, 'w') as f:
        json.dump({
            'maxcyc': int(maxcyc2),
        }, f, indent=2)


def run(min_dir):
    try:
        os.makedirs(min_dir)
    except OSError:
        pass

    pwd = os.getcwd()
    os.chdir(min_dir)
    topo, st1 = run_min1()
    run_min2(topo, st1)
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


def run_min1():
    system, model_pdb = load_model()
    pdb = app.PDBFile(model_pdb)

    with open('min1.in', 'r') as f:
        cntrl = json.load(f)

    # add force (restraints)
    res_weight = float(cntrl['restraint_wt']) * 418.4  # kcal/mol/A^2 to kJ/mol/nm^2
    force = mm.CustomExternalForce('k*(periodicdistance(x,y,z,x0,y0,z0)^2)')
    force.addGlobalParameter('k', res_weight)
    force.addPerParticleParameter('x0')
    force.addPerParticleParameter('y0')
    force.addPerParticleParameter('z0')
    topo = pdb.topology
    pos = pdb.positions
    for atom in topo.atoms():
        resname = atom.residue.name
        # atmname = atom.name
        if resname not in ('HOH', 'NA', 'CL'):
            ind = atom.index
            x0, y0, z0 = pos[ind]
            force.addParticle(ind, (x0, y0, z0))
    system.addForce(force)

    integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, 2.0 * unit.femtoseconds)
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
    simulation.minimizeEnergy(maxIterations=int(cntrl['maxcyc']))

    st = simulation.context.getState(getPositions=True, getEnergy=True)
    with open('min1.out', 'w') as f:
        print('# Epot', file=f)
        print('{:.4f}'.format(st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)), file=f)

    with open('min1.pdb', 'w') as f:
        app.PDBFile.writeFile(topo, st.getPositions(), f)

    # center and image the pdb file
    t = mdt.load('min1.pdb')
    t.image_molecules(inplace=True)
    t.center_coordinates(mass_weighted=True)
    with open('min1.pdb', 'w') as f:
        app.PDBFile.writeFile(topo, t.xyz[0] * 10.0, f)

    return topo, st


def run_min2(topo, st1):
    system, model_pdb = load_model()
    pdb = app.PDBFile(model_pdb)

    with open('min2.in', 'r') as f:
        cntrl = json.load(f)

    integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, 2.0 * unit.femtoseconds)
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

    simulation.context.setPositions(st1.getPositions())
    simulation.context.setPeriodicBoxVectors(*st1.getPeriodicBoxVectors())
    simulation.minimizeEnergy(maxIterations=int(cntrl['maxcyc']))

    st = simulation.context.getState(getPositions=True, getEnergy=True)
    with open('min2.out', 'w') as f:
        print('# Epot', file=f)
        print('{:.4f}'.format(st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)), file=f)

    pos = st.getPositions()
    bv = st.getPeriodicBoxVectors()
    np.savez('min2.npz', Positions=pos, PeriodicBoxVectors=bv)

    with open('min2.pdb', 'w') as f:
        app.PDBFile.writeFile(topo, pos, f)

    # center and image the pdb file
    t = mdt.load('min2.pdb')
    t.image_molecules(inplace=True)
    t.center_coordinates(mass_weighted=True)
    with open('min2.pdb', 'w') as f:
        app.PDBFile.writeFile(topo, t.xyz[0] * 10.0, f)


def submit_batch(work_dir, maxcyc1, maxcyc2, dependency=None):
    min_dir = os.path.join(work_dir, 'min')
    stdout = os.path.join(min_dir, 'stdout')
    stderr = os.path.join(min_dir, 'stderr')

    cmd = [
        '/usr/local/bin/sbatch',
        '--output', stdout,
        '--error', stderr,
        '--nodes', '1',
        '--time', '168:00:00',
        '--job-name', 'MIN',
        '--ntasks', '1',
        '--gres', 'gpu:1',
        '--partition', 'prowave',
    ]

    if dependency:
        cmd += ['--dependency', dependency]

    cmd += [
        os.path.abspath(__file__),
        work_dir,
        '--maxcyc1', maxcyc1,
        '--maxcyc2', maxcyc2,
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    submit_msg = proc.stdout.readline().decode()
    if submit_msg.startswith('Submitted batch job'):
        batch_jobid = int(submit_msg.split()[3])

        with open(os.path.join(min_dir, 'status'), 'w') as f:
            f.write('submitted %d' % batch_jobid)
        
        return batch_jobid
    else:
        return False


if __name__ == '__main__':
    args = PARSER.parse_args()
    _params = {
        'work_dir': args.work_dir,
        'maxcyc1': args.maxcyc1 if args.maxcyc1 else '1000',
        'maxcyc2': args.maxcyc2 if args.maxcyc2 else '2500',
    }
    write_min_input(_params)

    if args.batch:
        if args.dependency:
            _params['dependency'] = args.dependency
        result = submit_batch(**_params)
        print(result)
    else:
        run(os.path.join(args.work_dir, 'min'))
