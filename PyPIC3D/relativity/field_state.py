"""Conversions at the boundary of the densitized static-GR solver state.

Runtime slots 0/1/2 contain sqrt(gamma) times D/B/J at their native Yee
locations. Slot 7 contains the previous densitized D/B. External fields,
rho, phi, and the metric retain their existing conventions. File inputs and
outputs and particle forces use physical contravariant vectors.
"""


def densitize_vector(vector, samples):
    """Multiply each component by its own native metric volume factor."""
    return tuple(value * sample.sqrt_gamma for value, sample in zip(vector, samples))


def physical_vector(vector, samples):
    """Recover a physical contravariant vector from its native density."""
    return tuple(value / sample.sqrt_gamma for value, sample in zip(vector, samples))


def _convert_fields(fields, convert):
    D, B, J, rho, phi, external, metric, previous, overflow = fields
    return (convert(D, metric.D), convert(B, metric.B), convert(J, metric.D),
            rho, phi, external, metric,
            (convert(previous[0], metric.D), convert(previous[1], metric.B)),
            overflow)


def densitize_fields(fields):
    """Convert a physical initial/input state once into runtime storage."""
    return _convert_fields(fields, densitize_vector)


def physical_fields(fields):
    """Return a physical view of runtime state for diagnostics and output."""
    return _convert_fields(fields, physical_vector)
