import os
import numpy as np
import toml
import jax
import jax.numpy as jnp


METRIC_MINKOWSKI = 0
METRIC_CYLINDRICAL = 1
METRIC_STATIC = 2


def build_metric_from_parameters(simulation_parameters):
    """
    Build a metric descriptor used by the relativistic metric-aware pusher.

    Supported simulation parameters:
      - metric: "minkowski", "cylindrical", or "static"
      - metric_file: path to .npy/.npz/.toml static metric tensor (3x3 or 4x4)
      - metric_regularization: small positive value for coordinate singularities
    """
    metric_name = str(simulation_parameters.get("metric", "minkowski")).lower()
    regularization = float(simulation_parameters.get("metric_regularization", 1e-12))

    if metric_name == "minkowski":
        full_cov = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0], dtype=jnp.float64))
        spatial_cov = jnp.eye(3, dtype=jnp.float64)
        spatial_contra = jnp.eye(3, dtype=jnp.float64)
        lapse = jnp.asarray(1.0, dtype=jnp.float64)
        shift_cov = jnp.zeros(3, dtype=jnp.float64)
        shift_contra = jnp.zeros(3, dtype=jnp.float64)
        metric_type = METRIC_MINKOWSKI
    elif metric_name == "cylindrical":
        # Coordinate-dependent metric terms are evaluated at runtime.
        full_cov = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0], dtype=jnp.float64))
        spatial_cov = jnp.eye(3, dtype=jnp.float64)
        spatial_contra = jnp.eye(3, dtype=jnp.float64)
        lapse = jnp.asarray(1.0, dtype=jnp.float64)
        shift_cov = jnp.zeros(3, dtype=jnp.float64)
        shift_contra = jnp.zeros(3, dtype=jnp.float64)
        metric_type = METRIC_CYLINDRICAL
    elif metric_name in ("static", "file", "user_defined"):
        metric_file = simulation_parameters.get("metric_file", None)
        if metric_file is None:
            raise ValueError("metric='static' requires simulation_parameters.metric_file")
        full_cov, spatial_cov = _load_static_metric_tensor(metric_file)
        spatial_contra = jnp.linalg.inv(spatial_cov)
        lapse, shift_cov, shift_contra = _decompose_3plus1(full_cov, spatial_cov, spatial_contra)
        metric_type = METRIC_STATIC
    else:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. Expected one of: minkowski, cylindrical, static."
        )

    return {
        "metric_type": jnp.asarray(metric_type, dtype=jnp.int32),
        "spatial_cov": jnp.asarray(spatial_cov, dtype=jnp.float64),
        "spatial_contra": jnp.asarray(spatial_contra, dtype=jnp.float64),
        "full_cov": jnp.asarray(full_cov, dtype=jnp.float64),
        "lapse": jnp.asarray(lapse, dtype=jnp.float64),
        "shift_cov": jnp.asarray(shift_cov, dtype=jnp.float64),
        "shift_contra": jnp.asarray(shift_contra, dtype=jnp.float64),
        "regularization": jnp.asarray(regularization, dtype=jnp.float64),
        "is_metric_enabled": jnp.asarray(metric_type != METRIC_MINKOWSKI),
    }


def _load_static_metric_tensor(metric_file):
    if not os.path.exists(metric_file):
        raise FileNotFoundError(f"Metric file not found: {metric_file}")

    if metric_file.endswith(".npy"):
        metric = np.load(metric_file)
    elif metric_file.endswith(".npz"):
        data = np.load(metric_file)
        key = "metric" if "metric" in data else data.files[0]
        metric = data[key]
    elif metric_file.endswith(".toml"):
        cfg = toml.load(metric_file)
        if "metric" in cfg:
            metric = np.asarray(cfg["metric"])
        elif "spatial_metric" in cfg:
            metric = np.asarray(cfg["spatial_metric"])
        else:
            raise ValueError("TOML metric file must contain [metric] or [spatial_metric].")
    else:
        raise ValueError("Unsupported metric file format. Use .npy, .npz, or .toml")

    metric = jnp.asarray(metric, dtype=jnp.float64)
    full_metric = None
    if metric.shape == (4, 4):
        full_metric = metric
        metric = metric[1:, 1:]
    if metric.shape != (3, 3):
        raise ValueError(f"Static metric tensor must be shape (3,3) or (4,4), got {metric.shape}")

    metric = 0.5 * (metric + metric.T)
    det = jnp.linalg.det(metric)
    if jnp.abs(det) < 1e-20:
        raise ValueError("Static metric tensor is singular or near-singular.")

    if full_metric is None:
        full_metric = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0], dtype=jnp.float64))
        full_metric = full_metric.at[1:, 1:].set(metric)
    else:
        full_metric = 0.5 * (full_metric + full_metric.T)

    return full_metric, metric


def _decompose_3plus1(full_cov, spatial_cov, spatial_contra):
    # g00 = -alpha^2 + beta_i beta^i and g0i = beta_i
    g00 = full_cov[0, 0]
    shift_cov = full_cov[0, 1:]
    shift_contra = spatial_contra @ shift_cov
    beta_sq = shift_cov @ shift_contra
    alpha_sq = jnp.maximum(beta_sq - g00, 1e-20)
    lapse = jnp.sqrt(alpha_sq)
    return lapse, shift_cov, shift_contra


@jax.jit
def metric_terms_at_position(x, y, z, metric):
    metric_type = metric["metric_type"]
    dtype = jnp.result_type(x, y, z, metric["spatial_cov"])

    def minkowski_or_static(_):
        g_cov = metric["spatial_cov"].astype(dtype)
        g_contra = metric["spatial_contra"].astype(dtype)
        lapse = metric["lapse"].astype(dtype)
        shift_cov = metric["shift_cov"].astype(dtype)
        shift_contra = metric["shift_contra"].astype(dtype)
        gamma = jnp.zeros((3, 3, 3), dtype=dtype)
        return g_cov, g_contra, gamma, lapse, shift_cov, shift_contra

    def cylindrical(_):
        # Coordinates are interpreted as (r, phi, z).
        r = jnp.maximum(jnp.abs(x), metric["regularization"].astype(dtype))
        r2 = r * r
        inv_r = 1.0 / r
        inv_r2 = 1.0 / r2

        g_cov = jnp.array(
            [[1.0, 0.0, 0.0], [0.0, r2, 0.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        )
        g_contra = jnp.array(
            [[1.0, 0.0, 0.0], [0.0, inv_r2, 0.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        )

        gamma = jnp.zeros((3, 3, 3), dtype=dtype)
        gamma = gamma.at[0, 1, 1].set(-r)
        gamma = gamma.at[1, 0, 1].set(inv_r)
        gamma = gamma.at[1, 1, 0].set(inv_r)
        lapse = jnp.asarray(1.0, dtype=dtype)
        shift_cov = jnp.zeros(3, dtype=dtype)
        shift_contra = jnp.zeros(3, dtype=dtype)
        return g_cov, g_contra, gamma, lapse, shift_cov, shift_contra

    return jax.lax.switch(
        metric_type,
        (minkowski_or_static, cylindrical, minkowski_or_static),
        operand=None,
    )


@jax.jit
def geodesic_acceleration(v, x, metric):
    g_cov, _, christoffel, _, _, _ = metric_terms_at_position(x[0], x[1], x[2], metric)
    _ = g_cov
    return jnp.einsum("ijk,j,k->i", christoffel, v, v)


@jax.jit
def relativistic_metric_rhs(v, x, efield, bfield, q, m, constants, metric):
    """
    Returns dv/dt from Lorentz force plus the geodesic contribution:
        dv^i/dt = (q/m/gamma) * (E + v x B)^i - Gamma^i_{jk} v^j v^k
    """
    c = constants["C"]
    g_cov, _, christoffel, _, _, _ = metric_terms_at_position(x[0], x[1], x[2], metric)

    v2 = v @ (g_cov @ v)
    safety = jnp.maximum(1.0 - v2 / (c * c), 1e-16)
    gamma = 1.0 / jnp.sqrt(safety)

    lorentz = (q / (m * gamma)) * (efield + jnp.cross(v, bfield))
    geo = jnp.einsum("ijk,j,k->i", christoffel, v, v)
    return lorentz - geo


@jax.jit
def manufactured_geodesic_residual(v, accel, x, source, metric):
    """
    MMS residual for the geometric part of the equations:
      residual = dv/dt + Gamma(v,v) - source
    """
    geo = geodesic_acceleration(v, x, metric)
    return accel + geo - source


@jax.jit
def constitutive_from_metric_single_point(d_vec, b_vec, x, metric):
    """
    ENTITY 3+1 constitutive form:
      E = alpha * gamma^{-1} D + beta x B
      H = alpha * gamma^{-1} B - beta x D
    """
    _, g_contra, _, lapse, _, shift = metric_terms_at_position(x[0], x[1], x[2], metric)
    e_vec = lapse * (g_contra @ d_vec) + jnp.cross(shift, b_vec)
    h_vec = lapse * (g_contra @ b_vec) - jnp.cross(shift, d_vec)
    return e_vec, h_vec
