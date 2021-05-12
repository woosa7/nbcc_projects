import os
import sys
import json
import shutil
import gzip
import tempfile
import numpy as np
import mdtraj as mdt
from simtk import openmm as mm, unit
from simtk.openmm import app
from simtk.openmm.app.internal.pdbstructure import PdbStructure

# ---------------------------------------------------------------------
# contant values
dt_eq = 0.002
ntpr_eq = 2500
ntwx_eq = 2500


class equilibration:
    def __init__(self, workdir, chain_code, error_log):
        self.work_dir = workdir
        self.eq_dir = os.path.join(workdir, 'eq')
        try:
            os.makedirs(self.eq_dir)
        except OSError:
            pass

        self.chain_code = chain_code
        self.error_log = error_log


    def run_eq1(self, system, model_pdb):
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

        integrator = mm.LangevinIntegrator(float(cntrl['temp0']) * unit.kelvin,
                                           1.0 / unit.picoseconds,
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
            simulation.reporters.append(app.StateDataReporter(eq1_out, int(cntrl['ntpr']),
                                                              step=True, time=True,
                                                              potentialEnergy=True, kineticEnergy=True,
                                                              totalEnergy=True, temperature=True,
                                                              volume=True, density=True,
                                                              progress=True, remainingTime=True,
                                                              speed=True, totalSteps=nstlim,
                                                              separator=' '))
            try:
                simulation.step(nstlim)
            except Exception as e:
                print(e)
                with open(self.error_log, "a") as f:
                    print('{} --- equilibration : {}'.format(self.chain_code, e), file=f)
                return e, 'Exception'


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


    def run_eq2(self, system, model_pdb, topo, st1):
        pdb = app.PDBFile(model_pdb)

        # read eq2.in
        with open('eq2.in', 'r') as f:
            cntrl = json.load(f)

        temp0 = float(cntrl['temp0']) * unit.kelvin
        integrator = mm.LangevinIntegrator(temp0,
                                           1.0 / unit.picoseconds,
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
            simulation.reporters.append(app.StateDataReporter(eq2_out, int(cntrl['ntpr']),
                                                              step=True, time=True,
                                                              potentialEnergy=True, kineticEnergy=True,
                                                              totalEnergy=True, temperature=True,
                                                              volume=True, density=True,
                                                              progress=True, remainingTime=True,
                                                              speed=True, totalSteps=nstlim,
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

        return os.path.join(self.eq_dir, 'eq2.pdb')


    def run_equilibration(self, nstlim_eq_1, nstlim_eq_2, tempi, temp0, ntb_eq, restraint_wt):
        print('')
        print('---------------- run_equilibration')

        cwd = os.getcwd()
        os.chdir(self.eq_dir)

        # --------------------------------------------------------------------
        # Load model
        with open('../prep/model.xml', 'r') as f:
            system = mm.XmlSerializer.deserialize(f.read())

        model_pdb = PdbStructure(gzip.open('../prep/model_solv.pdb.gz', 'rt'))

        # --------------------------------------------------------------------
        # create input files
        with open('eq1.in', 'w') as f:
            json.dump({
                'dt': dt_eq,
                'nstlim': nstlim_eq_1,
                'tempi': tempi,
                'temp0': temp0,
                'ntb': ntb_eq,
                'ntwx': ntwx_eq,
                'ntpr': ntpr_eq,
                'restraint_wt': restraint_wt,
            }, f, indent=2)

        with open('eq2.in', 'w') as f:
            json.dump({
                'dt': dt_eq,
                'nstlim': nstlim_eq_2,
                'tempi': tempi,
                'temp0': temp0,
                'ntb': ntb_eq,
                'ntwx': ntwx_eq,
                'ntpr':
                ntpr_eq,
            }, f, indent=2)

        # --------------------------------------------------------------------
        # equilibration step
        topology, eq_state1 = self.run_eq1(system, model_pdb)

        if eq_state1 == 'Exception':
            return eq_state1

        eq_result = self.run_eq2(system, model_pdb, topology, eq_state1)

        os.chdir(cwd)

        return eq_result
