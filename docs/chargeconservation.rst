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

Fixed-metric schemes
--------------------

The ``static_metric`` solver evolves the contravariant densities ``D^i`` and
``B^i``, and both of its deposition schemes work with the *conformal* current
``sqrt(gamma) J^i``, returning the physical ``J^i`` that
``update_D_relativity`` consumes.

``GR_direct_deposition`` is a metric-weighted volume deposit. It samples the
lapse, shift and inverse metric at each particle to form ``alpha v^i - beta^i``,
and supports current filtering. It does not satisfy the discrete continuity
equation, so Gauss's law violation accumulates over a run.

``GR_esirkepov`` is charge conserving. It works because the conformal charge
density carries no metric,

.. math::

   \sqrt{\gamma}\,\rho = q\,S(x) / (\Delta x\,\Delta y\,\Delta z),

so the conformal continuity equation

.. math::

   \partial_t(\sqrt{\gamma}\rho) + \partial_i(\sqrt{\gamma}J^i) = 0

is the flat Esirkepov identity verbatim, and the same density decomposition
applies unchanged. The two schemes therefore share one metric-free kernel,
``esirkepov_tile_currents``.

Two things differ from the flat path. The new position must be supplied
explicitly rather than predicted as ``x + u*dt``, because ``particles.u`` stores
covariant ``u_i`` and that shortcut does not hold in a curved chart; the time
loop keeps the pre-push positions for this. And the out-of-plane component on a
reduced axis uses the displacement ``(x^{n+1} - x^n)/dt``, which is the
coordinate velocity the position update actually produced -- so the kernel needs
no metric interpolation at particle positions at all.

Because the backward-difference divergence of the backward-difference curl in
``update_D_relativity`` vanishes identically, satisfying discrete continuity
preserves

.. math::

   \partial_i(\sqrt{\gamma}D^i) = 4\pi\sqrt{\gamma}\rho

to round-off with no divergence cleaning. Configure it with:

.. code-block:: toml

   solver = "static_metric"
   current_calculation = "GR_esirkepov"
   particle_pusher = "hybrid_boris_geodesic"
   filter_j = "none"

Current filtering is rejected for ``GR_esirkepov`` for the same reason as the
flat scheme. ``GR_direct_deposition`` remains the default for
``static_metric``.

Reference
---------

Esirkepov, T. Z. (2001). Exact charge conservation scheme for particle-in-cell
simulation with an arbitrary form-factor. *Computer Physics Communications*,
135(2), 144-153.
