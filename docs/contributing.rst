Contributing
============

Change Checklist
----------------

- Existing tests pass and new behavior has focused coverage.
- Timestep order, Yee staggering, guard ownership, and boundary behavior are
  unchanged unless the pull request explicitly changes them.
- Tile dimensions, device mesh, and fixed particle capacity remain compatible.
- New configuration keys are consumed by initialization and documented.
- Output changes preserve species names, mesh coordinates, and tile offsets. 