"""
Outputs simulation trajectories in CRD format
"""
from __future__ import absolute_import

from crdfile import CRDFile

import simtk.openmm as mm
from simtk.unit import nanometer

class CRDReporter(object):
    """CRDReporter outputs a series of frames from a Simulation to a CRD file.

    To use it, create a CRDReporter, then add it to the Simulation's list of reporters.
    """

    def __init__(self, file, reportInterval, append=False, enforcePeriodicBox=None):
        """Create a CRDReporter.

        Parameters
        ----------
        file : string
            The file to write to
        reportInterval : int
            The interval (in time steps) at which to write frames
        append : bool=False
            If True, open an existing CRD file to append to.  If False, create a new file.
        enforcePeriodicBox: bool
            Specifies whether particle positions should be translated so the center of every molecule
            lies in the same periodic box.  If None (the default), it will automatically decide whether
            to translate molecules based on whether the system being simulated uses periodic boundary
            conditions.
        """
        self._reportInterval = reportInterval
        self._append = append
        self._enforcePeriodicBox = enforcePeriodicBox
        if append:
            mode = 'r'  # r+b
        else:
            mode = 'w'  # wb
        self._out = open(file, mode)
        self._crd = None


    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        tuple
            A six element tuple. The first element is the number of steps
            until the next report. The next four elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.  The final element specifies whether
            positions should be wrapped to lie in a single periodic box.
        """
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, False, False, False, self._enforcePeriodicBox)


    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """

        if self._crd is None:
            self._crd = CRDFile(
                self._out, simulation.topology, simulation.currentStep,
                self._reportInterval, self._append
            )

        self._crd.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())


    def __del__(self):
        self._out.close()
