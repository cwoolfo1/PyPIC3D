Development Guide
=================

Local Setup
-----------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -e .

Run Tests
---------

Run the focused implementation tests:

.. code-block:: bash

   python -m unittest tests/code_tests/*.py

Run numerical and convergence tests separately:

.. code-block:: bash

   python -m unittest tests/physics_tests/*.py

Distributed tile tests need enough JAX devices for their mesh. For example:

.. code-block:: bash

   XLA_FLAGS=--xla_force_host_platform_device_count=8 \
     python -m unittest tests/code_tests/distributed_ghost_cells_test.py

Build Docs
----------

.. code-block:: bash

   sphinx-autobuild docs _build/html --port 8008

This command watches for changes in the source files and rebuilds 
the docs automatically. Open a browser to http://localhost:8008 
to view the docs.

Debugging Tips
--------------

- Start with one tile covering the complete domain to separate numerical
  behavior from cross-device communication.
- For a multi-tile failure, verify tile divisibility, exposed device count, 
  number of ghost cells, and particle capacity.