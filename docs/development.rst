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

Distributed tile tests need one JAX device per tile. ``tests/__init__.py``
configures this before any test runs: 64-bit floats, the CPU platform, and 16
CPU devices. Setting ``JAX_PLATFORMS`` or ``JAX_NUM_CPU_DEVICES`` yourself
overrides the defaults, for example to run a single-tile test on a GPU:

.. code-block:: bash

   JAX_PLATFORMS=cuda python -m unittest tests/code_tests/yee_test.py

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