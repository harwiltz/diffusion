import jax
import jax.numpy as jnp

from functools import partial, reduce

import diffusion.initializers as initializers

class ParticleCluster(object):
    def __init__(self, n_atoms, initialization=initializers.dirac, loc_low=-10, loc_high=10, seed=0):
        self.n_atoms = n_atoms
        self.key, sub = jax.random.split(jax.random.PRNGKey(seed))
        self.particles = initialization(sub, self.n_atoms)
        self.sources = []
        self.perturb = lambda x: x
        self.loc_low = loc_low
        self.loc_high = loc_high

    def with_diffusions(self, *args):
        self.sources = args
        self.perturb = jax.jit(
            lambda key, x, t, dt: reduce(lambda acc, pk: pk[0].perturb(pk[1], acc, t, dt) + acc,
                                         zip(args, jax.random.split(key, 1 + len(args))[1:]),
                                         x)
        )
        return self

    def update(self, t, dt):
        self.key, sub = jax.random.split(self.key)
        self.particles = self.perturb(sub, self.particles, t, dt).clip(self.loc_low, self.loc_high)

class Diffusion(object):
    def __init__(self):
        pass
    
    def perturb(self, key, particles, t=0., dt=1.):
        raise NotImplementedError

class ItoDiffusion(Diffusion):
    def __init__(self, mu_fn, sigma_fn):
        super(ItoDiffusion, self).__init__()
        self.mu_fn = mu_fn
        self.sigma_fn = sigma_fn

    def perturb(self, key, particles, t, dt=1.):
        vel = dt * self.mu_fn(particles, t)
        noise = dt * self.sigma_fn(particles, t) * jax.random.normal(key, shape=particles.shape)
        return vel + noise

class SimpleHeatDiffusion(ItoDiffusion):
    def __init__(self, scale=1.):
        super(SimpleHeatDiffusion, self).__init__(mu_fn=lambda x, t: 0.,
                                                  sigma_fn= lambda x, t: scale)

class OrnsteinUhlenbeckDiffusion(ItoDiffusion):
    def __init__(self, theta=1., loc=0., scale=1.):
        super(OrnsteinUhlenbeckDiffusion, self).__init__(mu_fn=lambda x, t: theta * (loc - x),
                                                         sigma_fn=lambda x, t: scale)
        self.loc = loc
        self.scale = scale
