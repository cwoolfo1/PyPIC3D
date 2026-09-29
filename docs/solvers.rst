Field Solvers
=============

PyPIC3D provides these solver names:

- ``electrodynamic_yee``
- ``electrostatic``
- ``static_metric``

Electrodynamic Yee Step
-----------------------

The electrodynamic timestep keeps both particles and fields tiled. Its order is:

1. Interpolate total electric and magnetic fields to particles and push
   velocity.
2. Deposit current:

   - Direct deposition advances position by ``dt/2``, deposits at the
     centered position, then completes the second ``dt/2``.
   - Esirkepov deposition uses the old and predicted new positions, advances by
     ``dt``.

3. Update ``B`` by a half timestep from the old ``E``.
4. Update ``E`` by a full timestep from the half-step ``B`` and deposited
   ``J``.
5. Update ``B`` by a second half timestep from the new ``E``.

The field equations are:

.. math::

   \mathbf{B}^{n+1/2} =
   \mathbf{B}^{n} - \frac{\Delta t}{2}\nabla\times\mathbf{E}^{n},

.. math::

   \mathbf{E}^{n+1} =
   \mathbf{E}^{n} + \Delta t\left(
   c^2\nabla\times\mathbf{B}^{n+1/2}
   - \frac{\mathbf{J}^{n+1/2}}{\epsilon_0}\right),

.. math::

   \mathbf{B}^{n+1} =
   \mathbf{B}^{n+1/2}
   - \frac{\Delta t}{2}\nabla\times\mathbf{E}^{n+1}.

Electrostatic Step
------------------

The electrostatic timestep:

1. Pushes particle velocity from fields.
2. Advances position by ``dt``.
3. Deposits charge density ``rho``.
4. Solves Poisson in tiled storage with residual-controlled parallel Schwarz iterations.
5. Computes ``E = -grad(phi)``.

Each Schwarz iteration solves the owned potential on every tile with
tile-local conjugate gradient while holding ghost cells fixed.
Converged tiles stop updating through an active mask while other tile solves
continue. CG exits at ``electrostatic_local_cg_tol`` or
``electrostatic_local_cg_max_iterations`` and reduces only over each tile's
three owned spatial axes.

After each local solve, neighboring ``phi`` halos and physical conducting
boundaries are refreshed. The true Poisson residual is then evaluated in the
``guard_cells``-wide owned slabs next to tile interfaces. Schwarz exits at
``electrostatic_schwarz_tol`` or
``electrostatic_schwarz_max_iterations``. The previous timestep's potential is
the warm start, and an already-converged state performs no unnecessary solve.
No global potential, charge field, or Krylov reduction is assembled. The 
existing guard depth is the Schwarz overlap width.

Particle Pushers
----------------

``particle_pusher = "boris"`` selects Boris. With ``relativistic = true`` it
uses the relativistic Boris update; otherwise it uses the non-relativistic
form. ``particle_pusher = "higuera_cary"`` selects the Higuera-Cary relativistic
update.

Particles interpolate the sum of evolved and prescribed external fields.
Maxwell updates use only evolved fields.

Static-Metric Particle Sampling
--------------------------------

The ``static_metric`` solver uses ``hybrid_boris_geodesic``. Every particle
metric sample uses the shared ``interpolate_metric`` function: the velocity
update, position midpoint, direct GR current deposition, and particle-birth
momentum conversion all use the same reconstruction. Leapfrog initialization
also uses it through the shared pusher. ``GR_esirkepov`` deposits from particle
endpoints and does not need a separate particle-metric sample.

The sampler reconstructs lapse, shift, and the covariant spatial metric from
``YeeMetric.center`` on the base C grid using tensor-product cardinal cubic
Hermite polynomials (Catmull--Rom), with centered nodal slopes. It computes
the inverse and determinant from the reconstructed tensor rather than
interpolating the stored grid inverse or determinant. Analytic metric
providers supply grid values; there is no analytic or lower-order fallback
at particle positions and no interpolation-method setting.

Metric derivatives are derivatives of the same interpolant, including

.. math::

   \partial_k\gamma^{-1}
   = -\gamma^{-1}(\partial_k\gamma)\gamma^{-1}.

``ParticleMetric.grad_lapse[..., k]`` stores ``d_k alpha``,
``grad_shift[..., i, k]`` stores ``d_k beta^i``, and
``grad_gamma_inv[..., k, i, j]`` stores ``d_k gamma^ij``.
Sampling with ``derivatives=False`` returns the same metric values and
``None`` for the derivative fields. Smooth non-polynomial data generally
give third-order values and second-order derivatives; polynomial degree
alone does not imply fourth-order accuracy because the slopes are estimated.

The four-node stencil in each resolved direction requires at least three
guard cells for particle and midpoint sampling before tile migration.
Configurations using the GR particle sampler default to three and reject
smaller values. A globally single-cell direction uses its designated base
node and has zero metric derivative; a single-cell tile in a resolved
direction still interpolates normally.

Unused particle slots are sampled at safe interior positions. Samples outside
the available stencil produce NaNs, and checked execution reports invalid
active samples, nonpositive lapse, or invalid spatial tensors without repairs.
The derived volume factor preserves the chart's signed orientation in
reflected spherical and cylindrical guards.

This contract is independent of ``shape_factor``: electromagnetic field
gathering and particle deposition retain their selected particle shapes.
Output metadata identifies the reconstruction as
``cardinal_cubic_hermite_consistent_v1``; this is descriptive metadata, not
a selectable numerical mode.

Boundary Conditions and PML
---------------------------

The former ``polar`` boundary implementation and numeric boundary code 4 have
been removed. Spherical runs use explicit angular bounds, conducting fields,
and reflecting particles on a regular chart. Metric guard nodes must stay
away from the axes; the former exact-axis finite-volume mode is unsupported.

Field boundaries are set with ``x_bc``, ``y_bc``, and ``z_bc``:

- ``periodic``
- ``conducting``

Standard Yee conducting boundaries zero tangential electric components at
``g`` and ``g+n`` on the first and last tiles. The upper wall used to occupy
``g+n-1``; the effective cavity width is now ``N*dx`` instead of ``(N-1)*dx``.
Upper C endpoints are owned physical nodes even though stored in a halo slot;
neighbor exchange preserves them, including transverse communication.

For the static-metric finite-difference solver, ``conducting`` instead imposes
FIDO-field constraints on the evolved contravariant vectors:

.. math::

   G^i{}_j=n^i n_j, \qquad F^i{}_j=\delta^i{}_j-G^i{}_j,
   \qquad D^i\leftarrow G^i{}_jD^j,
   \qquad B^i\leftarrow F^i{}_jB^j.

At a coordinate face ``x^a=constant``, the spatial-metric unit normal is
``n_i=delta_i^a/sqrt(gamma^{aa})`` and
``n^i=gamma^{ia}/sqrt(gamma^{aa})``. A physically normal D can therefore have
nonzero coordinate-tangential components. Surface tensors are taken from the
existing C/V metric samples, with density-weighted component transfers to a
common location before projection. Reflections use ``2G-I`` for D and
``2F-I`` for B. At independent intersections D vanishes and B lies in the
common tangent space, defined by the metric Gram matrix. Exterior corner
reflections compose in x, y, z order.

These are local explicit projections: projector algebra is exact at the
reconstruction points; interpolating the stored staggered fields back to a
common surface introduces truncation error. There is no coupled boundary
solve. ``staggered.refresh_fields`` requires ``metric`` when applying a D/B
conducting boundary. Initialization, previous time levels, updated stages,
horizon/sponge processing, and temporary total fields for particle gathering
all use this adapter. Particle reflection and source folding are unchanged.

The auxiliary E/H routines remain pure constitutive calculations. Their
computed exterior values are preserved during internal halo exchange. The
FIDO constraints apply even with normal shift, where auxiliary tangential E
can be nonzero. Projection can change magnetic divergence and Gauss residuals
for incompatible fields; it does not model conductor surface charges or
currents.

The electrostatic solver extends potential constantly through conducting
exterior guards before taking its gradient.

Coordinate-stretched PML modifies the tile-local spatial derivatives before
the Ampere and Faraday curls are assembled. PML is supported only by
``electrodynamic_yee``. See :doc:`usage` for the TOML form.

Current Deposition
------------------

Select deposition with:

.. code-block:: toml

   current_calculation = "j_from_rhov"

or:

.. code-block:: toml

   current_calculation = "esirkepov"
   filter_j = "none"

See :doc:`chargeconservation` for deposition timing and filter behavior.
