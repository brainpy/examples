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
#     display_name: brainpy
#     language: python
#     name: brainpy
# ---

# %% [markdown]
# # [1D] Simple systems

# %%
import brainpy as bp

bp.math.enable_x64()
bp.math.set_platform('cpu')


# %% [markdown]
# ## Phase plane

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


# %%
analyzer = bp.analysis.PhasePlane1D(int_x,
                                    target_vars={'x': [-10, 10]},
                                    pars_update={'Iext': 0.})
analyzer.plot_vector_field()
analyzer.plot_fixed_point(show=True)

# %% [markdown]
# ## Codimension1

# %% [markdown]
# Then, create a bifurcation analyzer with ``bp.symbolic.Bifurcation``.

# %%
an = bp.analysis.Bifurcation1D(
  int_x,
  target_pars={'Iext': [-0.5, 0.5]},
  target_vars={"x": [-2, 2]},
  resolutions=0.001
)
an.plot_bifurcation(show=True)


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


# %% code_folding=[]
analyzer = bp.analysis.Bifurcation1D(
  int_x,
  target_pars={'lambda_': [-1, 4], 'mu': [-4, 4], },
  target_vars={'x': [-3, 3]},
  resolutions=0.1
)
analyzer.plot_bifurcation(show=True)
