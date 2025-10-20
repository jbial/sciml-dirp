from dataclasses import dataclass
from math import pi
from typing import Any

import diffrax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from sciml.utils import make_kgrids


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


@dataclass
class BurgersSolver(PsuedoSpectralSolver1D):
    nu: float = 0.1

    def __call__(self, t: float, uk: Float[Array, " dim"], args: Any | None = None) -> Float[Array, " dim"]:
        n_half = self.dimension // 2
        coeffs = uk[:n_half] + 1j * uk[n_half:]

        # Diffusion term: -ν * k² * uk (in Fourier space)
        diffusion_term = -self.nu * self.ks**2 * coeffs

        # Nonlinear term: -u * u_x (computed in spatial domain, then FFT)
        # Use 3/2 zero-padding de-aliasing for the quadratic product
        M = (3 * self.N) // 2
        padded = jnp.zeros(M // 2 + 1, dtype=coeffs.dtype)
        padded = padded.at[: self.N // 2 + 1].set(coeffs)

        ks_M = 2 * pi * jnp.fft.rfftfreq(M, self.L / M)
        u_M = jnp.fft.irfft(padded, n=M)
        ux_M = jnp.fft.irfft(1j * ks_M * padded, n=M)

        nonlinear_spatial_M = -u_M * ux_M
        nonlinear_M = jnp.fft.rfft(nonlinear_spatial_M, n=M)
        nonlinear_coeffs = nonlinear_M[: self.N // 2 + 1]

        flow = diffusion_term + nonlinear_coeffs
        return jnp.concatenate([jnp.real(flow), jnp.imag(flow)])


class KuramotoShivashinskySolver(PsuedoSpectralSolver1D):
    def __call__(self, t: float, uk: Float[Array, " dim"], args: Any | None = None) -> Float[Array, " dim"]:
        n_half = self.dimension // 2
        coeffs = uk[:n_half] + 1j * uk[n_half:]

        # Linear term: -u_xx - u_xxxx
        linear_term = -(self.ks**4 - self.ks**2) * coeffs

        # Nonlinear term: d/dx (0.5 * u**2)
        u = jnp.fft.irfft(coeffs, n=self.N)
        nonlinear_term = -1j * self.ks * jnp.fft.rfft(0.5 * u**2, n=self.N)

        flow = linear_term + nonlinear_term
        return jnp.concatenate([jnp.real(flow), jnp.imag(flow)])


class KortewegDeVriesSolver(PsuedoSpectralSolver1D):
    def __call__(self, t: float, uk: Float[Array, " dim"], args: Any | None = None) -> Float[Array, " dim"]:
        n_half = self.dimension // 2
        coeffs = uk[:n_half] + 1j * uk[n_half:]

        # Linear term: -u_xxx
        linear_term = 1j * self.ks**3 * coeffs

        # Nonlinear term: 3*(u^2)_x
        u = jnp.fft.irfft(coeffs, n=self.N)
        nonlinear_term = 3j * self.ks * jnp.fft.rfft(u**2, n=self.N)

        flow = linear_term + nonlinear_term
        return jnp.concatenate([jnp.real(flow), jnp.imag(flow)])


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
