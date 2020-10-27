"""
Saving Simulation Progress and Results

simulation.saveState()
- save the entire state of the simulation,
  including positions, velocities, box dimensions and much more in an XML file.

simulation.saveCheckpoint()
- save the entire simulation as a binary file.
- this binary can only be used to restart simulations on machines with the same hardware
  and the same OpenMM version as the one that saved it.

CheckpointReporter
- helpful in restarting simulations that failed unexpectedly or due to outside reasons.
"""

simulation.saveState('output.xml')
simulation.loadState('output.xml')


simulation.saveCheckpoint('state.chk')
simulation.loadCheckpoint('state.chk')


# CheckpointReporter
# save a checkpoint file every 5,000 steps
simulation.reporters.append(CheckpointReporter('checkpnt.chk', 5000))

# ---------------------------------------------------------------------------
