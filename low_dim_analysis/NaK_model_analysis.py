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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NaK model analysis

# %% [markdown]
# Here we will show you the neurodynamics analysis of a two-dimensional system model with the example of the $I_{\rm{Na,p+}}-I_K$ Model. 
#
# The dynamical system is given by:
# $$ C\dot{V} = I_{ext} - g_L * (V-E_L)-g_{Na}*m_\infty(V)(V-E_{Na})-g_K*n*(V-E_K)$$
#
# $$ \dot{n} = \frac{n_\infty(V)-n}{\tau(V)} $$
#
# where
#
# $$ m_\infty(V) = 1 \ / \ ({1+\exp(\frac{V_{\rm{n_{half}}}-V}{k_m})}) $$
#
# $$ n_\infty(V) = 1 \ / \ ({1+\exp(\frac{V_{\rm{n_{half}}}-V}{k_n})}) $$
#
# This model specifies a leak current $I_L$, persistent sodium current $I_{\rm{Na, p}}$ 
# with instantaneous activation kinetic, and a relatively slower persistent 
# potassium current $I_K$ with either high or low threshold (the two choices 
# result in fundamentally different dynamics). 

# %%
import brainpy as bp
import brainpy.math as bm

bm.set_dt(dt=0.02)

# %%
C = 1
E_L = -78  # different from high-threshold model
g_L = 8
g_Na = 20
g_K = 10
E_K = -90
E_Na = 60
Vm_half = -20
k_m = 15
Vn_half = -45  # different from high-threshold model
k_n = 5
tau = 1


@bp.odeint
def integral(V, n, t, Iext):
  m_inf = 1 / (1 + bm.exp((Vm_half - V) / k_m))
  I_leak = g_L * (V - E_L)
  I_Na = g_Na * m_inf * (V - E_Na)
  I_K = g_K * n * (V - E_K)
  dvdt = (-I_leak - I_Na - I_K + Iext) / C

  n_inf = 1 / (1 + bm.exp((Vn_half - V) / k_n))
  dndt = (n_inf - n) / tau

  return dvdt, dndt


# %% [markdown]
# ### Phase plane analysis

# %%
# analyzer = bp.symbolic.PhasePlane(
#   integral,
#   target_vars=dict(n=[0., 1.], V=[-90, 20], ),
#   pars_update={'Iext': 50.},
# )
# analyzer.plot_nullcline()
# analyzer.plot_vector_field()
# analyzer.plot_fixed_point()
# analyzer.plot_trajectory([{'V': -10, 'n': 0.2}, {'V': -80, 'n': 0.4}],
#                          duration=100.,
#                          show=True)

# %% [markdown]
# ### Codimension 1 bifurcation analysis

# %% [markdown]
# Here we show the codimension 1 bifurcation analysis of the $I_{\rm{Na,p+}}-I_K$ Model, in which $I_{ext}$ is varied in [0., 50.].

# %%
analyzer = bp.symbolic.Bifurcation(
  integral,
  target_pars={'Iext': [0, 50.]},
  target_vars={"V": [-90., 20.], 'n': [0., 1.],  },
  numerical_resolution={'Iext': 1., 'V': 1., 'n': 0.1},
  options={'sympy_solver_timeout': 20}
)
analyzer.plot_bifurcation(show=True)

# %% [markdown]
# ### Codimension 2 bifurcation analysis

# %% [markdown]
# Codimension 2 bifurcation analysis of the $I_{\rm{Na,p+}}-I_K$ Model, in which $I_{ext}$ is varied in [0., 50.], and "Vn_half" is varied in [-50, -40].

# %%
analyzer = bp.symbolic.Bifurcation(
  integral,
  target_pars={'Iext': [0, 50.], 'Vn_half': [-50, -40]},
  target_vars={'n': [0., 1.], "V": [-90., 20.], },
  numerical_resolution=0.1)

analyzer.plot_bifurcation(show=True)

# %% [markdown]
# ### Reference

# %% [markdown]
# 1. Izhikevich, Eugene M. Dynamical systems in neuroscience (Chapter 4). MIT press, 2007.
