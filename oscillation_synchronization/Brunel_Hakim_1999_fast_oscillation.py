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
# # _(Brunel & Hakim, 1999)_ Fast Global Oscillation

# %% [markdown]
# Implementation of the paper:
#
# - Brunel, Nicolas, and Vincent Hakim. "Fast global oscillations in networks of integrate-and-fire neurons with low firing rates." Neural computation 11.7 (1999): 1621-1671.
#
# Author: [Chaoming Wang](mailto:chao.brain@qq.com)

# %%
import brainpy as bp
import brainpy.math as bm

bm.set_platform('cpu')

# %%
Vr = 10.  # mV
theta = 20.  # mV
tau = 20.  # ms
delta = 2.  # ms
taurefr = 2.  # ms
duration = 100.  # ms
J = .1  # mV
muext = 25.  # mV
sigmaext = 1.  # mV
C = 1000
N = 5000
sparseness = float(C) / N


# %%
class LIF(bp.NeuGroup):
  def __init__(self, size, **kwargs):
    super(LIF, self).__init__(size, **kwargs)

    # variables
    self.V = bm.Variable(bm.ones(self.num) * Vr)
    self.t_last_spike = bm.Variable(-1e7 * bm.ones(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))

    # integration functions
    fv = lambda V, t: (-V + muext) / tau
    gv = lambda V, t: sigmaext / bm.sqrt(tau)
    self.int_v = bp.sdeint(f=fv, g=gv)

  def update(self, _t, _dt):
    V = self.int_v(self.V, _t, dt=_dt)
    in_ref = (_t - self.t_last_spike) < taurefr
    V = bm.where(in_ref, self.V, V)
    spike = V >= theta
    self.spike.value = spike
    self.V.value = bm.where(spike, Vr, V)
    self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
    self.refractory.value = bm.logical_or(in_ref, spike)


# %%
group = LIF(N)
syn = bp.models.DeltaSynapse(group, group, conn=bp.conn.FixedProb(sparseness),
                             delay=delta, post_has_ref=True, post_key='V', w=-J)
net = bp.Network(syn, group=group)

# %%
runner = bp.ReportRunner(net, monitors=['group.spike'], jit=True)
runner.run(duration)
bp.visualize.raster_plot(runner.mon.ts, runner.mon['group.spike'],
                         xlim=(0, duration), show=True)
