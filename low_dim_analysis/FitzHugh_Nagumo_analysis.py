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
# # FitzHugh-Nagumo analysis

# %% [markdown]
# The FitzHugh-Nagumo model is given by:
#
# $$ \frac {dV} {dt} = V(1 - \frac {V^2} 3) - w + I_{ext} $$
#
# $$ \tau \frac {dw} {dt} = V + a - b w $$

# %%
import brainpy as bp
import brainpy.math as bm
bm.set_dt(dt=0.02)


# %%
class FitzHughNagumo(bp.NeuGroup):
    def __init__(self, size, a=0.7, b=0.8, tau=12.5, Vth=1.9, **kwargs):
        super(FitzHughNagumo, self).__init__(size=size, **kwargs)

        self.a = a
        self.b = b
        self.tau = tau
        self.Vth = Vth

        self.V = bm.Variable(bm.zeros(size))
        self.w = bm.Variable(bm.zeros(size))
        self.spike = bm.Variable(bm.zeros(size))
        self.input = bm.Variable(bm.zeros(size))

    @staticmethod
    @bp.odeint(method='rk4')
    def integral(V, w, t, Iext, a, b, tau):
        dw = (V + a - b * w) / tau
        dV = V - V * V * V / 3 - w + Iext
        return dV, dw

    def update(self, _t, _dt):
        V, self.w[:] = self.integral(self.V, self.w, _t, self.input, self.a, self.b, self.tau)
        self.spike[:] = (V >= self.Vth) * (self.V < self.Vth)
        self.V[:] = V
        self.input[:] = 0.


# %% [markdown]
# ## Simulation

# %%
FNs = FitzHughNagumo(2, monitors=['V'])

FNs.run(duration=300., inputs=('input', 1.), report=True)
bp.visualize.line_plot(FNs.mon.ts, FNs.mon.V, show=True)

FNs.run(duration=(300., 600.), inputs=('input', 0.6), report=True)
bp.visualize.line_plot(FNs.mon.ts, FNs.mon.V, show=True)

# %% [markdown]
# ## Phase plane analysis

# %%
phase = bp.symbolic.OldPhasePlane(FNs,
                                  target_vars={'V': [-3, 2], 'w': [-2, 2]},
                                  fixed_vars=None,
                                  pars_update={'Iext': 1., "a": 0.7, 'b': 0.8, 'tau': 12.5})
phase.plot_nullcline()
phase.plot_fixed_point()
# phase.plot_trajectory(initials={'V': -1, 'w': 1}, duration=100.)
phase.plot_limit_cycle_by_sim(initials={'V': -1, 'w': 1}, duration=100.)
phase.plot_vector_field(show=True)

# %% [markdown]
# ## Codimension 1 bifurcation analysis

# %%
bifurcation = bp.symbolic.OldBifurcation(FNs,
                                         target_pars={'Iext': [-1, 1]},
                                         target_vars={'V': [-3, 2], 'w': [-2, 2]},
                                         fixed_vars=None,
                                         pars_update={'a': 0.7, 'b': 0.8, 'tau': 12.5},
                                         numerical_resolution=0.01)
bifurcation.plot_bifurcation(show=True)

# %% [markdown]
# ## Codimension 2 bifurcation analysis

# %%
bifurcation = bp.symbolic.OldBifurcation(FNs,
                                         target_pars=dict(a=[0.5, 1.], Iext=[0., 1.]),
                                         target_vars=dict(V=[-3, 3], w=[-3., 3.]),
                                         fixed_vars=None,
                                         pars_update={'b': 0.8, 'tau': 12.5},
                                         numerical_resolution=0.01)
bifurcation.plot_bifurcation(show=True)
