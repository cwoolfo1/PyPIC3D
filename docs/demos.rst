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

.. figure:: images/Ez_ot_shot.png
   :alt: Out-of-plane electric field from the Orszag-Tang vortex demo
   :align: center
   :width: 80%

   Orszag-Tang vortex out-of-plane electric field.

.. code-block:: bash

   cd demos/ot_vortex
   python initial_conditions.py
   PyPIC3D --config orszag_tang.toml

Still under development.

Harris-Sheet Reconnection
-------------------------

.. code-block:: bash

   cd demos/reconnection_2d
   python initial_data.py
   CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda \
     PyPIC3D --config harris_current.toml
   python analyze_data.py

The configuration uses two x-directed tiles and therefore needs two visible
JAX devices. The analysis command reads ``data/fields.pmd`` and writes
``analysis/field_lines.mp4``. The movie shows the full x domain and supported
cell-centered z extent, with normalized coordinates, time, and magnetic
magnitude. A labeled two-cell Gaussian display filter smooths the field lines;
the color scale stays fixed throughout the movie. Only the magnetic mesh is
required. Use ``--fields``, ``--output-dir``, ``--fps``, and ``--dpi`` to override
the input, output directory, frame rate (10), and resolution (150).

Blandford-Znajek Monopole
-------------------------

A monopole magnetosphere around a spinning Kerr black hole in
horizon-penetrating spherical Kerr-Schild coordinates, following Entity
Paper II (Section 4.5). Settings live in ``simulation_parameters.py``; the
runner takes no arguments.

.. code-block:: bash

   cd demos/static_metric_relativity/bz_monopole
   python run_bz_monopole.py
   python plot_entity_bz.py --data data --all

The run writes snapshots, ``diagnostics.npz`` and ``final_state.npz`` to
``data/`` and stops if the exterior Gauss or divergence-of-B residuals exceed
their tolerances. ``plot_entity_bz.py`` redraws the Figure 6 panels from the
saved snapshots.

Notes
-----

- Orszag-Tang and Harris-sheet runs load field and particle arrays generated
  by their ``initial_conditions.py`` and ``initial_data.py`` scripts.
- Output locations come from each demo's ``output_dir`` setting or default to
  the directory where the command is launched.
- A multi-tile demo needs one exposed JAX device per tile. See :doc:`tiling`
  before reducing the configured tile widths.
