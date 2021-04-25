import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from functools import partial
from matplotlib.animation import FuncAnimation

import diffusion.initializers as initializers

from diffusion.processes import ParticleCluster, Diffusion
from diffusion.processes import SimpleHeatDiffusion, OrnsteinUhlenbeckDiffusion, ItoDiffusion

INTERVAL = 50
DT = INTERVAL / 1000

def animate(i, cluster):
    plt.clf()
    cluster.update(i * DT, DT)
    plt.xlim(-4, 4)
    plt.hist(cluster.particles, bins=cluster.particles.shape[0] // 10 + 1, density=True)

def main(args):
    n_atoms = args.n_atoms
    if args.initialization == "uniform":
        initializer = partial(initializers.uniform, low=-10, high=10)
    elif args.initialization == "even":
        initializer = partial(initializers.even_space, low=-10, high=10)
    else:
        initializer = initializers.dirac

    base_cluster = ParticleCluster(n_atoms, initializer)

    if args.diffusion == "ornstein-uhlenbeck":
        cluster = base_cluster.with_diffusions(OrnsteinUhlenbeckDiffusion(theta=0.1))
    elif args.diffusion == "oscillator":
        cluster = base_cluster.with_diffusions(
            ItoDiffusion(mu_fn=lambda x, t: 1. * (3 * jnp.sin(t) - x),
                         sigma_fn=lambda x, t: 1.)
        )
    elif args.diffusion == "bimodal":
        loc = 2 * jnp.ones_like(base_cluster.particles).at[:n_atoms//2].mul(-1.)
        cluster = base_cluster.with_diffusions(
            OrnsteinUhlenbeckDiffusion(loc=loc, scale=2.)
        )
    else:
        cluster = ParticleCluster(n_atoms, initializer).with_diffusions(SimpleHeatDiffusion())

    fig, ax = plt.subplots()

    ani = FuncAnimation(fig, lambda i: animate(i, cluster), interval=INTERVAL)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion", type=str, default=None)
    parser.add_argument("--n_atoms", type=int, default=100)
    parser.add_argument("--initialization", type=str, default="dirac")
    args = parser.parse_args()
    main(args)
