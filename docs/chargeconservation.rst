Current Deposition
==================

PyPIC3D deposits current directly into tile-local Yee arrays. Contributions
that land in tile ghost cells are merged into neighboring tiles, followed by 
a ghost cell update.

Direct ``j_from_rhov``
----------------------

The direct method pushes particles to the centered position
``x + u*dt/2``, updates which tile they belong to, and deposits 
weighted charge times velocity on the staggered current component
grids. It then completes the second half of the position update.

Configure it with:

.. code-block:: toml

   current_calculation = "j_from_rhov"
   filter_j = "bilinear"

Supported filters are:

- ``none``: no current smoothing.
- ``bilinear``: tri-linear smoothing.
- ``digital``: nearest-neighbor digital filtering with coefficient ``alpha``.

The filter runs after ghost cell updates. Current ghosts are refreshed before and
after smoothing so each tile reads completed neighboring values. The same filter 
is applied to the evolved electric field used for interpolation so that the deposition 
and interpolation methods use the same stencil and remain consistent. Charge density 
uses the same digital filter when ``filter_j = "digital"``.

Esirkepov
---------

The Esirkepov path predicts the new position ``x + u*dt`` from the old
particle position, builds aligned old/new particle-shape stencils, and deposits
the charge-conserving current difference before particle ownership is
refreshed.

It supports shape factors 1 and 2 and reduced 1D/2D axes. Current filtering is
disabled for Esirkepov because the discrete continuity equation is satisfied
exactly. Configure it with:

.. code-block:: toml

   current_calculation = "esirkepov"
   filter_j = "none"

Initialization rejects Esirkepov combined with ``digital`` or ``bilinear``
current filtering so the discrete continuity construction is not silently
altered.

Reference
---------

Esirkepov, T. Z. (2001). Exact charge conservation scheme for particle-in-cell
simulation with an arbitrary form-factor. *Computer Physics Communications*,
135(2), 144-153.
