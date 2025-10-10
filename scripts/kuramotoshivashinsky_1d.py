from dataclasses import dataclass
from math import pi
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from jaxtyping import Float


@dataclass
class PsuedoSpectralSolver1D:
    N: int
    bounds: tuple[float, float] = (-1, 1)

    @property
    def L(self) -> float:
        return self.bounds[1] - self.bounds[0]

    @property
    def dimension(self) -> int:
        return self.N + 2

    def __post_init__(self) -> None:
        self.ks = 2 * pi * jnp.fft.rfftfreq(self.N, self.L / self.N)
        self.domain = jnp.linspace(*self.bounds, self.N, endpoint=False)

    def to_fourier(self, u: Float[Array, " N"], N: int | None = None) -> Float[Array, " dimension"]:
        """FFT of a signal u, returns real/imag split format

        :param u: input signal of shape (N,)
        :param N: dimension of input signal

        :return: Fourier coefficients in real/imag split format of shape (dimension)
        """
        if N is None:
            N = self.N

        # shape: (N//2 + 1,)
        uk = jnp.fft.rfft(u, n=N)

        return jnp.concatenate([uk.real, uk.imag], axis=-1)

    def to_spatial(self, uk: Float[Array, " dimension"], N: int | None = None) -> Float[Array, " N"]:
        """Inverse FFT of the modes to get u(x) at a certain time

        :param uk: array of flattened fourier coefficients (real and imag components), can have batch dimensions
        :param N: grid resolution in the spatial domain

        :return: solution in the spatial domain
        """
        if N is None:
            N = self.N

        coeffs = uk[..., : self.dimension // 2] + 1j * uk[..., self.dimension // 2 :]

        return jnp.fft.irfft(coeffs, n=N)

    def integrate(
        self,
        init_cond: Float[Array, " dim"],
        tspan: list[float, float],
        args: Any | None = None,
        num_save_pts: int | None = None,
        method: str = "Tsit5",
        max_dt: float = 1e-3,
        atol: float = 1e-8,
        rtol: float = 1e-8,
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
            rtol=rtol,  # error tolerance of solution
            atol=atol,  # error tolerance of solution
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


class KuramotoShivashinskySolver(PsuedoSpectralSolver1D):
    def __call__(self, t: float, uk: Float[Array, " dim"], args: Any | None = None) -> Float[Array, " dim"]:
        n_half = self.dimension // 2
        coeffs = uk[:n_half] + 1j * uk[n_half:]

        # Linear term: -u_xx - u_xxxx
        linear_term = -(self.ks**4 - self.ks**2) * coeffs

        # Nonlinear term: d/dx (0.5 * u**2)
        u = jnp.fft.irfft(coeffs)
        nonlinear_term = 1j * self.ks * jnp.fft.rfft(0.5 * u**2)

        flow = linear_term + nonlinear_term
        return jnp.concatenate([flow.real, flow.imag])


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    import imageio_ffmpeg
    import matplotlib as mpl

    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    solver = KuramotoShivashinskySolver(N=256, bounds=(0, 100))

    # make random fourier series as initial condition
    num_sines = 5
    rng = jax.random.key(0)
    amp_rng, freq_rng = jax.random.split(rng, 2)
    amplitudes = jax.random.normal(amp_rng, shape=num_sines)
    frequencies = jax.random.uniform(freq_rng, shape=num_sines, minval=0, maxval=2)
    ic = amplitudes @ jnp.sin(frequencies[:, jnp.newaxis] * solver.domain[jnp.newaxis, :])
    ic *= jnp.exp(solver.domain / solver.L)

    ic_k = solver.to_fourier(ic)
    tspan = (0, 100)
    ts, uks = solver.integrate(ic_k, tspan=tspan, num_save_pts=400, rtol=1e-5, atol=1e-7)

    us = solver.to_spatial(uks)

    extent = [tspan[0], tspan[1], solver.bounds[0], solver.bounds[1]]
    plt.figure(figsize=(10, 5))
    plt.imshow(us.T, aspect="auto", origin="lower", extent=extent, cmap="magma")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("KS equation")
    plt.colorbar(label="u")
    plt.savefig("figures/ks_1d.png")
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
    animation.save("figures/ks_1d.mp4")
    plt.close(fig)
