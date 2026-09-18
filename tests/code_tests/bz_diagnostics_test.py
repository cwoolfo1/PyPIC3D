"""BZ measurements retain normalization and expose polar extrema."""
import numpy as np
from demos.bz_monopole.run_bz_monopole import comparison_errors
from demos.bz_monopole.simulation_parameters import SimulationParameters


def test_analytic_profiles_and_polar_screening_are_reported_separately():
    p = SimulationParameters()
    r = np.linspace(1., 10., 64)
    theta = np.linspace(0., np.pi, 65)
    shape = (len(r), len(theta))
    snapshot = dict(r=r, theta=theta, time=200.,
                    Hphi=np.broadcast_to(-p.B0*p.spin*np.sin(theta)**2/8, shape),
                    omega=np.full(shape, .5*p.omega_h),
                    luminosity=np.full(len(r), 1.03*p.B0**2*p.omega_h**2/6),
                    D2_over_B2=np.full(shape, .05), DdotB_over_B2=np.zeros(shape))
    snapshot['D2_over_B2'][12, 0] = 2.
    measured = comparison_errors(snapshot, p)
    np.testing.assert_allclose(measured['luminosity_mean'], 1.03)
    np.testing.assert_allclose(measured['luminosity_relative_rms'], .03)
    for profile in measured['profiles']:
        assert profile['H_relative_l2'] < 1e-14
        assert profile['omega_relative_rms'] < 1e-14
    assert measured['bulk']['D2_over_B2_maximum'] == .05
    assert measured['full_exterior']['D2_over_B2_maximum'] == 2.
