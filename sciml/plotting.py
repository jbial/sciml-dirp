from pathlib import Path

import matplotlib.pyplot as plt
from celluloid import Camera
from IPython.display import HTML
from jaxtyping import Array, Float


def plot_solution(
    u: Float[Array, "T N"],
    tspan: tuple[float, float],
    bounds: tuple[float, float],
    title: str | None = None,
    cmap: str = "magma",
    save_path: str | Path | None = None,
) -> None:
    extent = [tspan[0], tspan[1], bounds[0], bounds[1]]
    plt.figure(figsize=(10, 5))
    plt.imshow(u.T, aspect="auto", origin="lower", extent=extent, cmap=cmap)
    plt.xlabel("$t$")
    plt.ylabel("$x$")
    plt.title(title)
    plt.colorbar(label="$u(x, t)$")

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_wavefront_1d(
    ts: Float[Array, " T"],
    xs: Float[Array, " N"],
    u: Float[Array, "T N"],
    bounds: tuple[float, float],
    vline: float | None = None,
    figsize: tuple[int, int] = (8, 4),
    save_path: str | Path | None = None,
) -> HTML:
    fig, ax = plt.subplots(figsize=figsize)
    camera = Camera(fig)

    for i in range(u.shape[0]):
        ax.plot(xs, u[i], color="r")
        if vline is not None:
            ax.axvline(x=vline, color="k", linestyle="-.")
        ax.set_ylim(u.min() - u.std(), u.max() + u.std())
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u(x, t)$")
        ax.text(0.5, 1.02, f"$t = {ts[i]:.2f}$", transform=ax.transAxes, ha="center", fontsize=12)
        camera.snap()

    animation = camera.animate(interval=50)
    plt.close(fig)

    if save_path is not None:
        animation.save(save_path)

    return HTML(animation.to_jshtml())
