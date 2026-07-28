Demos
=====

Runnable examples live under ``demos/``. Run commands from the repository root
unless a demo first generates local ``.npy`` initial conditions.

Two-Stream Instability
----------------------

.. figure:: images/two_stream_vortex.png
   :alt: 1D phase-space density of two counter-streaming electron beams
   :align: center
   :width: 80%

   two-stream instability phase-space diagram.



The two-stream instability demo uses one tile:

.. code-block:: bash

   PyPIC3D --config demos/two_stream/two_stream.toml

Weibel Instability
------------------

.. figure:: images/B_y_heatmap.png
   :alt: 1D magnetic field over time from the Weibel instability demo
   :align: center
   :width: 80%

   Weibel instability magnetic field evolution.

.. code-block:: bash

   PyPIC3D --config demos/weibel/weibel.toml

Orszag-Tang Vortex
------------------

.. figure:: images/magnetic_field_mag_ot.png
   :alt: Normalized magnetic field magnitude from the Orszag-Tang vortex demo
   :align: center
   :width: 80%

   Orszag-Tang vortex normalized magnetic field magnitude.

.. code-block:: bash

   cd demos/ot_vortex
   python initial_conditions.py
   PyPIC3D --config orszag_tang.toml

Still under development.

Harris-Sheet Reconnection
-------------------------

.. code-block:: bash

   cd demos/reconnection_2d
   python initial_conditions.py
   PyPIC3D --config harris_current.toml

Still under development.

Notes
-----

- Orszag-Tang and Harris-sheet runs load field and particle arrays generated
  by their ``initial_conditions.py`` scripts.
- Output locations come from each demo's ``output_dir`` setting or default to
  the directory where the command is launched.
- A multi-tile demo needs one exposed JAX device per tile. See :doc:`tiling`
  before reducing the configured tile widths.
