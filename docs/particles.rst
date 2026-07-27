Particle Species
================

Each ``[particleX]`` TOML section defines one species. Multiple species share
the same tiled mesh and global particle boundary conditions.

Required Fields
---------------

Each species defines:

- ``name``
- ``charge``
- ``mass``
- either ``N_particles`` or ``N_per_cell``

Example
-------

.. code-block:: toml

   [particle1]
   name = "electrons"
   N_particles = 30000
   charge = -1.602e-19
   mass = 9.1093837e-31
   temperature = 293000
   number_density = 1.0e15

Initialization Options
----------------------

- Thermal state: ``temperature`` or ``vth``, with optional directional
  temperatures ``Tx``, ``Ty``, and ``Tz``.
- Statistical weight: ``weight`` directly, or ``number_density`` to derive the
  weight from particles per cell and cell volume.
- Sampling bounds: ``xmin/xmax``, ``ymin/ymax``, and ``zmin/zmax``. Defaults
  are the complete centered simulation domain.
- Initial arrays or scalar offsets: ``initial_x``, ``initial_y``,
  ``initial_z``, ``initial_vx``, ``initial_vy``, and ``initial_vz``. String
  values are loaded from ``.npy`` files.
- Update controls: ``update_pos`` and ``update_v`` plus component switches
  ``update_x/y/z`` and ``update_vx/vy/vz``.

Without external arrays, positions are sampled uniformly within the species
bounds and velocities are sampled from the configured thermal distributions.
A scalar initial position places particles within a sub-cell interval around
that coordinate when the axis contains more than one cell. A scalar initial
velocity is added to the sampled thermal velocity.

Particle Boundary Conditions
----------------------------

The runtime particle boundaries are global simulation parameters:

- ``particle_x_bc``
- ``particle_y_bc``
- ``particle_z_bc``

Supported values are:

- ``periodic``: wrap positions across the domain.
- ``reflecting``: reflect position and the normal velocity.
- ``absorbing``: mark particles inactive after they leave the domain.

Inactive particles retain allocated slots for static JAX shapes but no longer
move, push, deposit, or contribute to particle diagnostics.

Tiled Particle State
--------------------

``TiledParticles`` contains position ``x``, velocity ``u``, and active-mask
arrays with leading tile and species axes. ``SpeciesConfig`` separately stores
charge, mass, weight, and update masks once per species. This avoids repeating
constant metadata in every particle slot.

Particle slots have fixed capacity after initialization. Retiling moves active
particles between adjacent tile owners; insufficient destination capacity is a
hard runtime error. See :doc:`tiling` for exact shapes and capacity selection.

Shape Factors
-------------

``simulation_parameters.shape_factor`` controls interpolation and deposition:

- ``1``: first-order particle shapes.
- ``2``: second-order particle shapes.

Use the same shape factor when comparing runs because it changes the particle
stencil, numerical smoothing, and guard-cell reach.
