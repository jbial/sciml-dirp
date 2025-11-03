import os
from multiprocessing import cpu_count

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpu_count()}"

from collections.abc import Callable, Iterator
from math import pi
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from chex import dataclass
from einops import einsum, rearrange
from equinox.nn import Linear
from jaxtyping import Array, ArrayLike, Float
from tqdm import tqdm

from sciml.psuedospectral import BurgersSolver

ActivationFunction = Callable[[Float[Array, "... dim"]], Float[Array, "... dim"]]
Key = ArrayLike


def swish(x: Float[Array, "..."]) -> Float[Array, "..."]:
    return x / (1 + jnp.exp(-x))


class SpectralConv1D(eqx.Module):
    weight: Float[Array, "n_modes in_channels out_channels"]
    modes: int

    def __init__(
        self, key: Key, in_channels: int, out_channels: int, modes: int, weight_scale: float = 0.1
    ) -> "SpectralConv1D":
        real_key, imag_key = jax.random.split(key)
        w_real = jax.random.normal(real_key, shape=(modes, in_channels, out_channels)) * weight_scale
        w_real = jnp.clip(w_real, min=-2 * weight_scale, max=2 * weight_scale)
        w_imag = jax.random.normal(imag_key, shape=(modes, in_channels, out_channels)) * weight_scale
        w_imag = jnp.clip(w_imag, min=-2 * weight_scale, max=2 * weight_scale)
        self.weight = w_real + 1j * w_imag
        self.modes = modes

    def __call__(self, x: Float[Array, "B N C_in"]) -> Float[Array, "B N C_out"]:
        B, N, _ = x.shape
        x_fft = jnp.fft.rfft(x, axis=1)  # (B, N//2 + 1, C_in)
        _, Nk, _ = x_fft.shape
        modes = min(self.modes, Nk)
        out_modes = jnp.zeros((B, Nk, self.weight.shape[-1]), dtype=x_fft.dtype)
        x_fft_trunc = x_fft[:, :modes, :]  # (B, modes, Cin)
        w = self.weight[:modes, :, :]  # (modes, Cin, Cout)
        transformed_modes = einsum(x_fft_trunc, w, "B modes C_in, modes C_in C_out -> B modes C_out")
        out_modes = out_modes.at[:, :modes, :].set(transformed_modes)
        y = jnp.fft.irfft(out_modes, n=N, axis=1)
        return y


class FNOBlock(eqx.Module):
    spectral: SpectralConv1D
    w: Linear
    activation: ActivationFunction

    def __init__(self, key: Key, width: int, modes: int, activation: ActivationFunction = swish) -> "FNOBlock":
        spec_key, w_key = jax.random.split(key)
        self.spectral = SpectralConv1D(spec_key, width, width, modes)
        self.w = Linear(width, width, key=w_key)
        self.activation = activation

    def __call__(self, x: Float[Array, "B N C"]) -> Float[Array, "B N C"]:
        y = self.spectral(x) + jax.vmap(jax.vmap(self.w))(x)
        return self.activation(y)


class FNO1D(eqx.Module):
    lift: Linear
    blocks: list[FNOBlock]
    proj: Linear

    def __init__(
        self,
        key: Key,
        in_channels: int,
        out_channels: int,
        width: int,
        depth: int,
        modes: int,
        activation: ActivationFunction = swish,
    ) -> "FNO1D":
        keys = jax.random.split(key, num=2 + depth)
        self.lift = Linear(in_channels, width, key=keys[0])
        self.blocks = [FNOBlock(k, width, modes, activation) for k in keys[2:]]
        self.proj = Linear(width, out_channels, key=keys[1])

    def __call__(self, x: Float[Array, "B N C_in"]) -> Float[Array, "B N C_out"]:
        h = jax.vmap(jax.vmap(self.lift))(x)
        for blk in self.blocks:
            h = blk(h)
        return jax.vmap(jax.vmap(self.proj))(h)


@dataclass
class FourierSeriesInitialConditionSampler:
    L: float
    domain: Float[Array, " N"]
    key: Key
    num_modes_range: tuple[int, int] = (2, 8)
    freq_range: tuple[float, float] = (2 * pi, 6 * pi)
    amplitudes_range: tuple[float, float] = (0.1, 2.0)

    def _sample_waveform(self, key: Key) -> Float[Array, " N"]:
        modes_key, freqs_key, amps_key = jax.random.split(key, num=3)
        M = max(self.num_modes_range)
        n_modes = jax.random.randint(
            modes_key, shape=(), minval=self.num_modes_range[0], maxval=self.num_modes_range[1]
        )
        mask = jnp.arange(M) < n_modes
        freqs = jax.random.uniform(freqs_key, shape=(M,), minval=self.freq_range[0], maxval=self.freq_range[1])
        freqs = freqs * mask
        amplitudes = jax.random.uniform(
            amps_key, shape=(M,), minval=self.amplitudes_range[0], maxval=self.amplitudes_range[1]
        )
        amplitudes = jnp.sort(amplitudes, axis=-1, descending=True) * mask
        modes = einsum(freqs / self.L, self.domain, "n_modes, N -> n_modes N")
        waveform = einsum(amplitudes, jnp.sin(modes), "n_modes, n_modes N -> N")
        return waveform

    def __call__(self, n_samples: int = 1) -> Float[Array, "n_samples N"]:
        self.key, key_batch = jax.random.split(self.key, num=2)
        sample_keys = jax.random.split(key_batch, num=n_samples)
        return jax.vmap(self._sample_waveform)(sample_keys)


def dataloader(ys: Float[Array, "M T N"], batch_size: int, history_size: int) -> Iterator[Float[Array, "batch_size 1"]]:
    M, _, T = ys.shape
    inds = np.arange(M)
    while True:
        perm = np.random.permutation(inds)
        start = 0
        end = batch_size
        while end <= ys.shape[0]:
            tinds = np.random.choice(T - history_size - 1)
            batch_idx = perm[start:end]
            batch = ys[batch_idx, :, tinds : tinds + history_size + 1]
            yield batch[..., :history_size], batch[..., -1:]
            start = end
            end = start + batch_size


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    key = jax.random.key(1337)
    data_key, model_key = jax.random.split(key)

    @dataclass
    class BurgersTrainConfig:
        N: int = 256
        viscosity: float = 1e-3

        # data generation config
        num_train_ics: int = 32
        num_timepoints: int = 1024
        tspan: tuple[float, float] = (0.0, 100.0)
        num_chunks: int = 8

        # model config
        history_size: int = 2
        in_channels: int = 2
        out_channels: int = 1
        width: int = 64
        depth: int = 4
        n_modes: int = 32

        # train config
        iterations: int = 100
        batch_size: int = 32
        warmup_pct: float = 0.1
        log_interval: int = 50
        init_lr: float = 1e-6
        max_lr: float = 1e-3

        # eval config
        rollout_steps: int = 100
        num_eval_ics: int = 8
        num_ood_ics: int = 8

    cfg = BurgersTrainConfig()

    solver = BurgersSolver(N=cfg.N, nu=cfg.viscosity)
    xs = solver.domain  # (N,)

    # random fourier series as an initial condition
    ic_sampler = FourierSeriesInitialConditionSampler(L=solver.L, domain=xs, key=data_key)
    ics = ic_sampler(cfg.num_train_ics + cfg.num_eval_ics)

    def integrate_ics(
        ics: Float[Array, "num_ics N"],
        num_timepoints: int,
        tspan: tuple[float, float],
        num_chunks: int = 8,
        **integrator_kwargs: dict[str, Any],
    ) -> tuple[Float[Array, "num_ics T"], Float[Array, "num_ics T N"]]:
        num_ics, _ = ics.shape
        assert num_ics % num_chunks == 0, "Number of ICs must be divisible by number of chunks"
        sharded_ics = rearrange(
            ics, "(chunks chunksize) N -> chunks chunksize N", chunks=num_chunks, chunksize=num_ics // num_chunks
        )

        def _integrate_single_ic(ic: Float[Array, " N"]) -> tuple[Float[Array, " T"], Float[Array, "T N"]]:
            ic_k = solver.to_fourier(ic)
            ts, uks = solver.integrate(ic_k, tspan, num_save_pts=num_timepoints, **integrator_kwargs)
            return ts, solver.to_spatial(uks)

        chunk_integrator = jax.vmap(_integrate_single_ic)
        t_chunks, u_chunks = jax.pmap(chunk_integrator)(sharded_ics)
        ts = rearrange(t_chunks, "M C T -> (M C) T")
        us = rearrange(u_chunks, "M C T N -> (M C) T N")
        return ts, us

    # generate data
    # ts shape: (num_ics, num_timepoints)
    # trajectories shape: (num_ics, N, num_timepoints)
    data_path = "data/burgers_trajectories.npz"
    if os.path.exists(data_path):
        data = jnp.load(data_path)
        train_trajectories = data["train_trajectories"]
        eval_trajectories = data["eval_trajectories"]
        ood_trajectories = data["ood_trajectories"]
    else:
        ood_ic_sampler = FourierSeriesInitialConditionSampler(
            L=solver.L, domain=xs, key=data_key, amplitudes_range=(1.0, 3.0)
        )
        ood_ics = ic_sampler(cfg.num_ood_ics)

        _, trajectories = integrate_ics(
            ics, num_timepoints=cfg.num_timepoints, tspan=cfg.tspan, num_chunks=cfg.num_chunks
        )
        _, ood_trajectories = integrate_ics(ood_ics, num_timepoints=cfg.num_timepoints, tspan=cfg.tspan, num_chunks=1)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        jnp.savez(
            data_path,
            train_trajectories=trajectories[: cfg.num_train_ics],
            eval_trajectories=trajectories[cfg.num_train_ics :],
            ood_trajectories=ood_trajectories,
        )
    train_trajectories = rearrange(train_trajectories, "num_ics T N -> num_ics N T")
    eval_trajectories = rearrange(eval_trajectories, "num_ics T N -> num_ics N T")
    ood_trajectories = rearrange(ood_trajectories, "num_ics T N -> num_ics N T")

    # define model
    fno = FNO1D(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        width=cfg.width,
        depth=cfg.depth,
        modes=cfg.n_modes,
        key=model_key,
    )

    # loss and optimizer
    @eqx.filter_value_and_grad
    def loss_fn(model: eqx.Module, u: Float[Array, "B C_in N"], u_tp1: Float[Array, "B C_out N"]) -> Float[Array, ""]:
        preds = model(u)
        return jnp.mean((preds - u_tp1) ** 2)

    warmup_steps = int(cfg.warmup_pct * cfg.iterations)
    decay_steps = cfg.iterations - warmup_steps

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=cfg.init_lr, peak_value=cfg.max_lr, warmup_steps=warmup_steps, decay_steps=decay_steps
    )
    optimizer = optax.adamw(learning_rate=lr_schedule)
    opt_state = optimizer.init(eqx.filter(fno, eqx.is_array))

    @eqx.filter_jit
    def step_fn(
        model: eqx.Module, batch: tuple[Float[Array, "B C_in N"], Float[Array, "B C_out N"]], opt_state: Any
    ) -> tuple[Float[Array, ""], eqx.Module, Any]:
        x, y = batch
        loss, grads = loss_fn(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    # training
    loader = dataloader(train_trajectories, batch_size=cfg.batch_size, history_size=cfg.history_size)
    with tqdm(total=cfg.iterations, desc="Training FNO", dynamic_ncols=True, leave=True) as pbar:
        for i in range(cfg.iterations):
            t_batch = next(loader)
            loss, fno, opt_state = step_fn(fno, t_batch, opt_state)
            loss = float(loss)
            if i % cfg.log_interval == 0:
                tqdm.write(f"Iteration: {i:4d} | MSE: {loss:12.5f}")
            pbar.update(1)

    # evaluation: new ICs, per-step NRMSE, heatmaps, and videos
    def rollout(model: eqx.Module, ic: Float[Array, "N C_in"], steps: int) -> Float[Array, "steps N C_out"]:
        hist = ic[jnp.newaxis]
        outs = [ic]
        for _ in range(steps):
            x = hist
            y = model(x)  # (1, N, C_out)
            outs.append(y.squeeze(0))
            hist = jnp.concatenate([x[..., 1:], y], axis=-1)
        return jnp.concatenate(outs, axis=-1)

    def nrmse(y_true: Float[Array, " N"], y_pred: Float[Array, " N"]) -> Float[Array, ""]:
        err = y_pred - y_true
        rmse_t = jnp.sqrt(jnp.mean(err * err))
        denom_t = jnp.std(y_true) + 1e-12
        return rmse_t / denom_t

    nrmse_curves = []
    eval_preds = []
    eval_ts = []
    preds = jax.vmap(rollout, in_axes=(None, 0, None))(fno, eval_trajectories[..., :2], cfg.rollout_steps)

    y_true = rearrange(eval_trajectories[..., : 2 + cfg.rollout_steps], "E N R -> E R N")
    y_pred = rearrange(preds, "E N R -> E R N")
    errors = jax.vmap(jax.vmap(nrmse))(y_true, y_pred).cumsum(axis=-1)

    mean_cumerror = errors.mean(axis=0)
    std_cumerror = errors.std(axis=0)

    plt.figure(figsize=(10, 4))
    steps = jnp.arange(2 + cfg.rollout_steps)
    plt.plot(steps, mean_cumerror)
    plt.fill_between(steps, mean_cumerror - std_cumerror, mean_cumerror + std_cumerror, alpha=0.2)
    plt.xlabel("rollout step")
    plt.ylabel("NRMSE")
    plt.title("Cumulative NRMSE")
    plt.show()
    plt.close()

    # single IC heatmap and rollout video
    idx = int(np.random.randint(0, preds.shape[0]))
    t0, t1 = float(cfg.tspan[0]), float(cfg.tspan[1])
    y0, y1 = float(xs[0]), float(xs[-1])

    plt.figure(figsize=(12, 4))
    im = plt.imshow(
        preds[idx],  # shape: (N, steps)
        aspect="auto",
        origin="lower",
        extent=[t0, t1, y0, y1],
        cmap="magma",
    )
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(f"FNO prediction — eval IC {idx}")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    # video of 1D waveform across rollout
    import imageio_ffmpeg
    import matplotlib as mpl
    from celluloid import Camera

    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    def save_video(u_seq: Array, xs_arr: Array, title: str, path: str, frame_interval: int = 150) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        camera = Camera(fig)
        y_min = float(u_seq.min() - u_seq.std())
        y_max = float(u_seq.max() + u_seq.std())
        for t in range(u_seq.shape[1]):
            ax.plot(xs_arr, u_seq[:, t], color="r")
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(float(xs_arr[0]), float(xs_arr[-1]))
            ax.set_xlabel("x")
            ax.set_ylabel("û(x,t)")
            ax.set_title(title)
            camera.snap()
        anim = camera.animate(interval=frame_interval)
        os.makedirs("figures", exist_ok=True)
        anim.save(path, fps=10)
        plt.close(fig)

    save_video(preds[idx], xs, f"FNO rollout — eval IC {idx}", f"figures/fno_eval_ic_{idx}.mp4")
