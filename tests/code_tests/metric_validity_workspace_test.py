"""Equivalent SPD validation without a particle-count-sized cuSolver workspace."""
import jax
import jax.numpy as jnp
import numpy as np
from PyPIC3D.relativity.interpolate_metric import positive_definite_3x3
jax.config.update('jax_enable_x64',True)


def test_sylvester_matches_eigenvalue_signs():
    rng=np.random.default_rng(519)
    q,_=np.linalg.qr(rng.normal(size=(2000,3,3)))
    values=np.exp(rng.uniform(-4,4,size=(2000,3)))
    values[500:1000,0]*=-1
    values[1000:1500,:2]*=-1
    values[1500:]*=-1
    tensors=np.einsum('nij,nj,nkj->nik',q,values,q)
    old=np.linalg.eigvalsh(tensors).min(-1)>0
    new=np.asarray(jax.jit(positive_definite_3x3)(jnp.asarray(tensors)))
    np.testing.assert_array_equal(new,old)
    edge=jnp.array([np.diag([1.,1.,0.]),np.diag([1.,1.,1e-20]),np.diag([1.,1.,-1e-20]),np.diag([1.,np.nan,1.])])
    np.testing.assert_array_equal(positive_definite_3x3(edge),[False,True,False,False])
    # Singular non-diagonal tensor is rejected without a floor or repair.
    assert not bool(positive_definite_3x3(jnp.ones((3,3))))
