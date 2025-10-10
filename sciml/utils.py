"""General Utility Functions"""

from math import pi

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


def make_kgrids(npoints: list[int], lengths: list[float]) -> Float[Array, "N ... N//2+1"]:
    freqs = []
    for N, L in zip(npoints[:-1], lengths[:-1]):
        fftfreqs = 2 * pi * jnp.fft.fftfreq(N, d=L / N)
        freqs.append(fftfreqs)
    freqs.append(2 * pi * jnp.fft.rfftfreq(npoints[-1], d=lengths[-1] / npoints[-1]))
    return jnp.meshgrid(*freqs, indexing="ij")
