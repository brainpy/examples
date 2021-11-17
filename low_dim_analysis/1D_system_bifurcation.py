# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: jax-cpu
#     language: python
#     name: jax-cpu
# ---

# %% [markdown]
# # 1D system birfurcation

# %%
import brainpy as bp


# %% [markdown]
# ## Codimension1

# %% [markdown]
# Here we will show the birfurcation analysis of 1D system with dummy test neuronal model.
#
# $$\dot{x} = x^3-x + I$$
#
# First, let's define the model.

# %%
@bp.odeint
def int_x(x, t, Iext):
    dx = x ** 3 - x + Iext
    return dx


# %% [markdown]
# Then, create a bifurcation analyzer with ``bp.symbolic.Bifurcation``.

# %%
an = bp.symbolic.Bifurcation(
    int_x,
    target_pars={'Iext': [-0.5, 0.5]},
    target_vars={"x": [-2, 2]},
    numerical_resolution=0.0001)

_ = an.plot_bifurcation(show=True)


# %% [markdown]
# ## Codimension2

# %% [markdown]
# Here we define the following 1D model for codimension 2 bifurcation testing.
#
# $$\dot{x} = \mu+ \lambda x - x^3$$

# %%
@bp.odeint
def int_x(x, t, mu, lambda_):
    dxdt = mu + lambda_ * x - x ** 3
    return dxdt


# %%
# please install numba!=0.54.x, because they have bugs

analyzer = bp.symbolic.Bifurcation(
    int_x,
    target_pars={'mu': [-4, 4], 'lambda_': [-1, 4]},
    target_vars={'x': [-3, 3]},
    numerical_resolution=0.1)
_ = analyzer.plot_bifurcation(show=True)
