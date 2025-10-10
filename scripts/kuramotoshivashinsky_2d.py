from dataclasses import dataclass
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from celluloid import Camera
from jax import Array
from jaxtyping import Float

from sciml.utils import make_kgrids


@dataclass
class PsuedoSpectralSolverND:
    N: tuple[int]
    bounds: list[tuple[float, float]]

    @property
    def D(self) -> int:
        return len(self.N)

    @property
    def L(self) -> list[float]:
        return [b[1] - b[0] for b in self.bounds]

    @property
    def dimension(self) -> list[int]:
        return int(jnp.prod(jnp.array(self.N[:-1]))) * (self.N[-1] + 2)

    def __post_init__(self) -> None:
        self.ks = jnp.array(make_kgrids(self.N, self.L))
        self.domain = jnp.mgrid[[slice(start, end, N * 1j) for (start, end), N in zip(self.bounds, self.N)]]

    def to_fourier(self, u: Float[Array, " N"], N: list[int] | None = None) -> Float[Array, " dimension"]:
        """FFT of a signal u, returns real/imag split format

        :param u: input signal of shape (N,)
        :param N: dimension of input signal

        :return: Fourier coefficients in real/imag split format of shape (dimension)
        """
        if N is None:
            N = self.N

        # shape: (N, ..., N, N//2 + 1)
        uk = jnp.fft.rfftn(u, s=N)

        return jnp.concatenate([uk.real, uk.imag], axis=-1)

    def to_spatial(self, uk: Float[Array, " dimension"], N: list[int] | None = None) -> Float[Array, " N"]:
        """Inverse FFT of the modes to get u(x) at a certain time

        :param uk: array of flattened fourier coefficients (real and imag components), can have batch dimensions
        :param N: grid resolution in the spatial domain

        :return: solution in the spatial domain
        """
        if N is None:
            N = self.N

        assert len(N) == self.D, "Number of sizes doesnt match dimensions"

        coeffs = uk[..., : N[-1] // 2 + 1] + 1j * uk[..., N[-1] // 2 + 1 :]

        return jnp.fft.irfftn(coeffs, s=N)

    def integrate(
        self,
        init_cond: Float[Array, " dim"],
        tspan: list[float, float],
        args: Any | None = None,
        num_save_pts: int | None = None,
        method: str = "Tsit5",
        max_dt: float = 1e-3,
        atol: float = 1e-5,
        rtol: float = 1e-7,
        pid: tuple[float] = [0.3, 0.3, 0.0],
        max_steps: int | None = None,
    ) -> tuple[Float[Array, " T"], Float[Array, "T dim"]]:
        """Integrate the dynamical system, and save a equispaced trajectory

        :param init_cond: initial condition for the system
        :param tspan: integration timespan [t_init, t_final]
        :param args: optional args
        :param num_save_pts: number of points to save for the trajectory from the integrator

        :returns: Tuple of integrator times, and solution trajectory
        """
        if num_save_pts is None:
            save_pts = diffrax.SaveAt(t1=True)
            progress_bar = diffrax.NoProgressMeter()
        else:
            save_pts = diffrax.SaveAt(ts=jnp.linspace(tspan[0], tspan[1], num_save_pts))
            progress_bar = diffrax.TqdmProgressMeter(num_save_pts)

        stepsize_controller = diffrax.PIDController(
            rtol=atol,  # error tolerance of solution
            atol=rtol,  # error tolerance of solution
            pcoeff=pid[0],  # proportional strength for PID stepsize controller
            icoeff=pid[1],  # integral strength for PID stepsize controller
            dcoeff=pid[2],  # integral strength for PID stepsize controller
            dtmax=max_dt,  # max step size
        )
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.__call__),
            getattr(diffrax, method)(),
            *tspan,
            y0=init_cond,
            args=args,
            dt0=max_dt,
            saveat=save_pts,
            max_steps=max_steps,
            stepsize_controller=stepsize_controller,
            progress_meter=progress_bar,
        )
        return jnp.asarray(sol.ts), jnp.asarray(sol.ys)


class KuramotoShivashinsky2DSolver(PsuedoSpectralSolverND):
    def __call__(self, t: float, uk: Float[Array, "D dim"], args: Any | None = None) -> Float[Array, "D dim"]:
        n_half = self.N[-1] // 2 + 1
        coeffs = uk[..., :n_half] + 1j * uk[..., n_half:]

        # linear (stiff) term
        nabla_2 = (self.ks**2).sum(axis=0)
        linear_term = -(nabla_2**2 + nabla_2) * coeffs

        # nonlinear term
        u_grad = 1j * self.ks * coeffs[jnp.newaxis, :]
        u = jnp.fft.irfftn(u_grad, s=self.N)
        u_2 = 0.5 * (u**2).sum(axis=0)
        nonlinear_term = jnp.fft.rfftn(u_2, s=self.N)

        flow = linear_term + nonlinear_term
        return jnp.concatenate([flow.real, flow.imag], axis=-1)


def main():
    jax.config.update("jax_enable_x64", True)

    import imageio_ffmpeg
    import matplotlib as mpl

    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    solver = KuramotoShivashinsky2DSolver(N=(32, 32), bounds=[(-1, 1), (-1, 1)])

    twosigma = 0.1
    ic = jnp.exp(-(jnp.linalg.norm(solver.domain, axis=0) ** 2) / twosigma)
    ic_k = solver.to_fourier(ic)

    ts, uks = solver.integrate(ic_k, tspan=(0, 1), num_save_pts=100)
    us = solver.to_spatial(uks)

    fig, ax = plt.subplots()
    camera = Camera(fig)
    for frame in us:
        im = ax.imshow(frame, cmap="magma", animated=True)
        camera.snap()
    animation = camera.animate()
    animation.save("figures/ks_2d.mp4")


if __name__ == "__main__":
    main()
