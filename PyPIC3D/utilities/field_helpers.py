def add_external_fields(E, B, external_fields):
    """
    Add prescribed external fields to the self-consistent fields.

    Maxwell updates should use E and B by themselves. Particle pushes and total
    energy diagnostics should use the returned totals, because those are the
    fields particles actually see.
    """
    external_E, external_B = external_fields
    total_E = tuple(e + ext_e for e, ext_e in zip(E, external_E))
    total_B = tuple(b + ext_b for b, ext_b in zip(B, external_B))
    return total_E, total_B
