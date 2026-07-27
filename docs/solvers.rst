Field Solvers
=============

PyPIC3D has two production solver names:

- ``electrodynamic_yee``
- ``electrostatic``

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

``E`` is digitally filtered after its update, and ``B`` is 
filtered only after the second half-step, preserving the 
current field time-centering. The filter coefficient is ``alpha``.

Electrostatic Step
------------------

The electrostatic timestep:

1. Pushes particle velocity from fields.
2. Advances position by ``dt``.
3. Deposits charge density ``rho``.
4. Assembles global ``rho`` for the finite-difference conjugate-gradient
   Poisson solve.
5. Computes ``E = -grad(phi)``.

The electrostatic runtime therefore uses tiled particles, ``rho``, ``phi``,
and ``E``, but the Poisson solve remains a deliberate global bridge. Thus,
initialization forces one tile covering the complete domain until a domain 
decomposition can be implemented for the Poisson solver.

Particle Pushers
----------------

``particle_pusher = "boris"`` selects Boris. With ``relativistic = true`` it
uses the relativistic Boris update; otherwise it uses the non-relativistic
form. ``particle_pusher = "higuera_cary"`` selects the Higuera-Cary relativistic
update.

Particles interpolate the sum of evolved and prescribed external fields.
Maxwell updates use only evolved fields.

Boundary Conditions and PML
---------------------------

Field boundaries are set with ``x_bc``, ``y_bc``, and ``z_bc``:

- ``periodic``
- ``conducting``

Conducting boundaries zero tangential electric components at the global wall.
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