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
dt_md = 0.002
ntpr_md = 2500

class production:
    def __init__(self, workdir, chain_code, error_log):
        self.work_dir = workdir
        self.md_dir = os.path.join(workdir, 'md')
        try:
            os.makedirs(self.md_dir)
        except OSError:
            pass

        self.chain_code = chain_code
        self.error_log = error_log


    def run_production(self, nstlim_md, target_temperature, ntb_md, pressure_md, ntwx_md, simulation_time_ns):
        print('')
        print('---------------- run_production')

        cwd = os.getcwd()
        os.chdir(self.md_dir)

        target_temp = float(target_temperature)

        # --------------------------------------------------------------------
        # Load model
        with open('../prep/model.xml', 'r') as f:
            system = mm.XmlSerializer.deserialize(f.read())

        model_pdb = PdbStructure(gzip.open('../prep/model_solv.pdb.gz', 'rt'))

        pdb = app.PDBFile(model_pdb)
        topo = pdb.topology

        # --------------------------------------------------------------------
        # md production
        md_result = ''

        for i in range(0, simulation_time_ns, 1):
            md_serial = i + 1
            if md_serial == 1:
                md_inp_file = '../eq/eq2.npz'
            else:
                md_inp_file = 'md_%s.npz' % (md_serial - 1)

            md_in = 'md_{}.in'.format(md_serial)
            md_out = 'md_{}.out'.format(md_serial)
            md_npz = 'md_{}.npz'.format(md_serial)
            md_dcd = 'md_{}.dcd'.format(md_serial)
            md_pdb = 'md_{}.pdb'.format(md_serial)

            if os.path.exists(md_pdb):
                md_result = os.path.join(self.md_dir, md_pdb)
                continue

            # create input files
            with open(md_in, 'w') as f:
                json.dump({
                    'dt': dt_md,
                    'nstlim': nstlim_md,
                    'temp0': target_temp,
                    'ntb': ntb_md,
                    "pressure": pressure_md,
                    'ntwx': ntwx_md,
                    'ntpr': ntpr_md,
                    'md_inp': md_inp_file,
                }, f, indent=2)

            rst = np.load(md_inp_file)
            pos = rst['Positions']

            temp0 = float(target_temp) * unit.kelvin
            integrator = mm.LangevinIntegrator(temp0,
                                               1.0 / unit.picoseconds,
                                               float(dt_md) * unit.picoseconds)
            integrator.setConstraintTolerance(0.00001)

            if int(ntb_md) == 2:
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

            simulation.reporters.append(app.DCDReporter(md_dcd, int(ntwx_md)))
            nstlim = int(nstlim_md)

            with open(md_out, 'w') as md_out:
                simulation.reporters.append(app.StateDataReporter(md_out, int(ntpr_md),
                                                                  step=True, time=True,
                                                                  potentialEnergy=True, kineticEnergy=True,
                                                                  totalEnergy=True, temperature=True, volume=True,
                                                                  density=True, progress=True,
                                                                  remainingTime=True, speed=True,
                                                                  totalSteps=nstlim,
                                                                  separator=' '))
                # simulation.step(nstlim)
                try:
                    simulation.step(nstlim)
                except Exception as e:
                    print(e)
                    with open(self.error_log, "a") as f:
                        print('{} --- production : {}'.format(self.chain_code, e), file=f)
                    return 'Exception'


            st = simulation.context.getState(getPositions=True, getVelocities=True)

            pos = st.getPositions()
            vel = st.getVelocities()
            bv = st.getPeriodicBoxVectors()
            np.savez(md_npz, Positions=pos, Velocities=vel, PeriodicBoxVectors=bv)

            with open(md_pdb, 'w') as f:
                app.PDBFile.writeFile(topo, pos, f)

            # center and image the pdb file
            t = mdt.load(md_pdb)
            t.image_molecules(inplace=True)
            t.center_coordinates(mass_weighted=True)
            with open(md_pdb, 'w') as f:
                app.PDBFile.writeFile(topo, t.xyz[0] * 10.0, f)

            md_result = os.path.join(self.md_dir, md_pdb)


        os.chdir(cwd)

        return md_result
