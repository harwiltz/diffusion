import jax
import jax.numpy as jnp

def dirac(key, n_atoms, loc=0):
    return loc * jnp.ones(n_atoms)

def even_space(key, n_atoms, low=-1, high=1):
    return jnp.linspace(low, high, n_atoms)

def uniform(key, n_atoms, low=-1, high=1):
    return jax.random.uniform(key, shape=(n_atoms,), minval=low, maxval=high)
