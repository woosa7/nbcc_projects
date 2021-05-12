import os
import sys
import gzip
import json
import numpy as np
import mdtraj as mdt
from simtk import openmm as mm, unit
from simtk.openmm import app
from simtk.openmm.app.internal.pdbstructure import PdbStructure


class minimization:
    def __init__(self, workdir, chain_code, error_log):
        self.work_dir = workdir
        self.min_dir = os.path.join(workdir, 'min')
        try:
            os.makedirs(self.min_dir)
        except OSError:
            pass

        self.chain_code = chain_code
        self.error_log = error_log


    def run_min1(self, system, model_pdb):
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

        temper = int(cntrl['temperature'])

        integrator = mm.LangevinIntegrator(temper * unit.kelvin,
                                           1.0 / unit.picoseconds,
                                           2.0 * unit.femtoseconds)
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

        try:
            simulation.minimizeEnergy(maxIterations=int(cntrl['maxcyc']))
        except Exception as e:
            print(e)
            with open(self.error_log, "a") as f:
                print('{} --- minimization : {}'.format(self.chain_code, e), file=f)
            return e, 'Exception'

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


    def run_min2(self, system, model_pdb, topo, st1):
        pdb = app.PDBFile(model_pdb)

        with open('min2.in', 'r') as f:
            cntrl = json.load(f)

        temper = int(cntrl['temperature'])

        integrator = mm.LangevinIntegrator(temper * unit.kelvin,
                                           1.0 / unit.picoseconds,
                                           2.0 * unit.femtoseconds)
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

        return os.path.join(self.min_dir, 'min2.pdb')


    def run_minimization(self, target_temperature, maxcyc_min_1, maxcyc_min_2, restraint_wt):
        print('')
        print('---------------- run_minimization')

        cwd = os.getcwd()
        os.chdir(self.min_dir)

        # --------------------------------------------------------------------
        # Load model
        with open('../prep/model.xml', 'r') as f:
            system = mm.XmlSerializer.deserialize(f.read())

        model_pdb = PdbStructure(gzip.open('../prep/model_solv.pdb.gz', 'rt'))

        # --------------------------------------------------------------------
        # create input files
        with open('min1.in', 'w') as f:
            json.dump({
                'restraint_wt': restraint_wt,
                'maxcyc': int(maxcyc_min_1),
                'temperature': int(target_temperature),
            }, f, indent=2)

        with open('min2.in', 'w') as f:
            json.dump({
                'maxcyc': int(maxcyc_min_2),
                'temperature': int(target_temperature),
            }, f, indent=2)

        # --------------------------------------------------------------------
        # minimization step
        topology, min_state1 = self.run_min1(system, model_pdb)

        if min_state1 == 'Exception':
            return min_state1

        min_result = self.run_min2(system, model_pdb, topology, min_state1)

        os.chdir(cwd)

        return min_result
