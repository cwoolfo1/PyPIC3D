Simulation Grids
================

PyPIC3D builds global coordinate lines and tile-local coordinate lines during
initialization. They are stored in ``DynamicParameters.grids``.

Grid Construction
-----------------

The physical domain is centered on zero, with widths ``x_wind``, ``y_wind``,
and ``z_wind``. The global coordinate tuples are:

- ``grids.center``: collocated/base coordinate lines with one exterior point
  on each side.
- ``grids.vertex``: half-cell-staggered coordinate lines with one exterior
  point on each side.

Electrostatic runs use collocated lines for both entries. Electrodynamic runs
also build ``grids.tiled_center_grid`` and ``grids.tiled_vertex_grid`` with the
configured tile shape and guard depth. See :doc:`tiling`.

Yee Staggering
--------------

Electric field and current components use the staggered grid along their
component axis and the collocated grid along transverse axes. Magnetic
components use the collocated grid along their component axis and staggered
grids along transverse axes:

.. code-block:: text

   Ex, Jx : (vertex_x, center_y, center_z)
   Ey, Jy : (center_x, vertex_y, center_z)
   Ez, Jz : (center_x, center_y, vertex_z)

   Bx     : (center_x, vertex_y, vertex_z)
   By     : (vertex_x, center_y, vertex_z)
   Bz     : (vertex_x, vertex_y, center_z)

.. image:: images/yeegrid.png
   :alt: Yee grid staggering
   :align: center

Boundary Encoding
-----------------

Field boundaries are converted during initialization to integer codes stored
in ``StaticParameters.boundary_conditions``:

- ``0``: periodic
- ``1``: conducting

Periodic ghost cells wrap across the global domain. Conducting ghost cells do 
not wrap, and the Yee update zeros tangential electric components on the 
physical wall. Charge, current, and fluid-moment folding use the particle 
boundary conditions.

Reduced Dimensions
------------------

Set an inactive dimension to one physical cell, for example ``Ny = 1`` for an
x-z simulation. PyPIC3D keeps the axis in every array and collapses its
deposition/interpolation stencil onto the physical cell. This preserves a
consistent 3D array contract for 1D, 2D, and 3D runs.
