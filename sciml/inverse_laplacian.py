
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import lineax as lx
import jax.random as jr

def create_A(N, dx):
    diag_main = -2 * np.ones(N)
    diag_main = diag_main.at[0].set(1)
    diag_main = diag_main.at[-1].set(1)
    diag_upper = np.ones(N-1)
    diag_lower = np.ones(N-1)

    A = lx.TridiagonalLinearOperator(
        diagonal=diag_main,
        upper_diagonal=diag_upper,
        lower_diagonal=diag_lower,
    )
    A = A / dx**2
    return A

def create_b(f, x, u0, u1):
    N = len(x)
    b = -f(x)
    b = b.at[0].add(u0)
    b = b.at[-1].add(u1)
    return b





