from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from jaxtyping import Float

from sciml.psuedospectral import PsuedoSpectralSolver1D


class KortewegDeVriesSolver(PsuedoSpectralSolver1D):
    def __call__(self, t: float, uk: Float[Array, " dim"], args: Any | None = None) -> Float[Array, " dim"]:
        n_half = self.dimension // 2
        coeffs = uk[:n_half] + 1j * uk[n_half:]

        # Linear term: -u_xxx
        linear_term = 1j * self.ks**3 * coeffs

        # Nonlinear term: 3*(u^2)_x
        u = jnp.fft.irfft(coeffs, n=self.N)
        nonlinear_term = -3j * self.ks * jnp.fft.rfft(u**2, n=self.N)
        mask = self.ks < 2 / 3 * jnp.max(self.ks)
        nonlinear_term = nonlinear_term * mask

        flow = linear_term + nonlinear_term
        return jnp.concatenate([jnp.real(flow), jnp.imag(flow)])


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    import imageio_ffmpeg
    import matplotlib as mpl

    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    solver = KortewegDeVriesSolver(N=256, bounds=(0, 50))

    twosigma = 5
    ic = -jnp.exp(-((solver.domain - 0.5 * sum(solver.bounds)) ** 2) / twosigma) + 0.5
    ic_k = solver.to_fourier(ic)

    tspan = (0, 100)
    ts, uks = solver.integrate(ic_k, tspan=tspan, num_save_pts=300)

    us = solver.to_spatial(uks)

    extent = [tspan[0], tspan[1], solver.bounds[0], solver.bounds[1]]
    plt.figure(figsize=(10, 5))
    plt.imshow(us.T, aspect="auto", origin="lower", extent=extent, cmap="magma")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("KdV equation")
    plt.colorbar(label="u")
    plt.savefig("figures/kdv.png")
    plt.close()

    from celluloid import Camera

    fig, ax = plt.subplots(figsize=(8, 4))
    camera = Camera(fig)

    for i in range(us.shape[0]):
        ax.plot(solver.domain, us[i], color="r")
        ax.set_ylim(us.min() - us.std(), us.max() + us.std())
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        camera.snap()

    animation = camera.animate(interval=50)
    animation.save("figures/kdv.mp4")
    plt.close(fig)
