Simulation Grids
================

PyPIC3D builds structured 3D grids during initialization and stores them in
``world['grids']``.

Grid Construction
-----------------

By default, initialization builds a Yee-style pair of grids:

- ``world['grids']['center']``: collocated/base index grid with one ghost cell
  on each side
- ``world['grids']['vertex']``: staggered half-cell grid with one ghost cell
  on each side

These are generated from ``Nx, Ny, Nz`` and the resolved coordinate bounds.
Old configs can continue to give centered widths with
``x_wind, y_wind, z_wind``. New configs can instead give explicit
``x_min/x_max``, ``y_min/y_max``, and ``z_min/z_max`` pairs. Once the grid is
constructed, the grid arrays are the source of truth for the physical domain.

The field components follow the legacy PyPIC3D Yee placement. Electric current
and electric field components use the vertex grid along their component axis
and the center grid along transverse axes. Magnetic field components use the
center grid along their component axis and the vertex grid along transverse
axes. For example, ``Ex`` and ``Jx`` live on
``(vertex_x, center_y, center_z)``, while ``Bx`` lives on
``(center_x, vertex_y, vertex_z)``.

.. image:: images/yeegrid.png
   :alt: Yee grid staggering
   :align: center

Boundary Encoding
-----------------

Field boundary conditions are stored in
``world['boundary_conditions']`` as integer codes for JAX-safe usage:

- ``0``: periodic
- ``1``: conducting

Dimensionality Handling
-----------------------

PyPIC3D supports effectively 1D/2D/3D runs by setting inactive dimensions to a
single cell (for example ``Ny = 1``). Deposition and solver routines infer
active dimensions from array shapes.
