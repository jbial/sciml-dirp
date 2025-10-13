from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from jaxtyping import Float

from sciml.psuedospectral import PsuedoSpectralSolverND


class KuramotoShivashinsky2DSolver(PsuedoSpectralSolverND):
    def __call__(self, t: float, uk: Float[Array, "D dim"], args: Any | None = None) -> Float[Array, "D dim"]:
        n_half = self.N[-1] // 2 + 1
        coeffs = uk[..., :n_half] + 1j * uk[..., n_half:]

        # linear (stiff) term: -∇⁴u + ∇²u
        nabla_2 = (self.ks**2).sum(axis=0)
        linear_term = -(nabla_2**2 - nabla_2) * coeffs

        # nonlinear term: -0.5*|∇u|²
        du_k = 1j * self.ks * coeffs[jnp.newaxis, :]
        du = jnp.fft.irfftn(du_k, s=self.N, axes=(1, 2))
        du_2 = 0.5 * (du**2).sum(axis=0)
        nonlinear_term = -jnp.fft.rfftn(du_2, s=self.N)

        flow = linear_term + nonlinear_term
        return jnp.concatenate([flow.real, flow.imag], axis=-1)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    import imageio_ffmpeg
    import matplotlib as mpl
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    half_L = 50
    resolution = 128
    solver = KuramotoShivashinsky2DSolver(N=(resolution, resolution), bounds=[(-half_L, half_L), (-half_L, half_L)])

    rng = jax.random.key(0)
    coeff_rng, loc_rng, bandwidth_rng = jax.random.split(rng, 3)

    # gaussian mixture initial condition
    modes = 1000
    mixture_weights = jax.random.dirichlet(coeff_rng, alpha=jnp.ones(modes))
    locations = jax.random.uniform(loc_rng, shape=(modes, 2), minval=-half_L, maxval=half_L)
    bandwidths = jax.random.uniform(bandwidth_rng, shape=(modes, 1, 1), minval=1e-4, maxval=1)
    ic = mixture_weights[:, jnp.newaxis, jnp.newaxis] * jnp.exp(
        -(jnp.linalg.norm(solver.domain[jnp.newaxis, ...] - locations[..., jnp.newaxis, jnp.newaxis], axis=1) ** 2)
        / bandwidths
    )
    ic = ic.sum(axis=0)

    ic_k = solver.to_fourier(ic)

    ts, uks = solver.integrate(ic_k, tspan=(0, 200), num_save_pts=600)
    us = solver.to_spatial(uks)

    fig, ax = plt.subplots()
    extent = [-half_L, half_L, -half_L, half_L]

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    im = ax.imshow(us[0], cmap="magma", extent=extent, origin="lower", aspect="auto")
    im.set_clim(us[0].min(), us[0].max())
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("u(x, y, t)")
    title = ax.set_title(f"2D Kuramoto-Shivashisky Equation (t = {ts[0]:.2f})")

    def update(frame_idx):
        frame = us[frame_idx]
        im.set_data(frame)
        im.set_clim(frame.min(), frame.max())
        title.set_text(f"2D Kuramoto-Shivashisky Equation (t = {ts[frame_idx]:.2f})")
        return [im, title]

    animation = FuncAnimation(fig, update, frames=len(us), blit=True, interval=100)
    writer = FFMpegWriter(fps=10)
    animation.save("figures/ks_2d.mp4", writer=writer)
