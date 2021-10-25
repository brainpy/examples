# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: brainmodels
#     language: python
#     name: brainmodels
# ---

# %% [markdown]
# # *(Brette, et, al., 2007)* CUBA

# %% [markdown]
# Implementation of the paper:
#
# - Brette, R., Rudolph, M., Carnevale, T., Hines, M., Beeman, D., Bower, J. M., et al. (2007), Simulation of networks of spiking neurons: a review of tools and strategies., J. Comput. Neurosci., 23, 3, 349–98
#
# which is based on the balanced network proposed by:
#
# - Vogels, T. P. and Abbott, L. F. (2005), Signal propagation and logic gating in networks of integrate-and-fire neurons., J. Neurosci., 25, 46, 10786–95
#
# Authors:
#
# - Chaoming Wang (chao.brain@qq.com)

# %%
import brainmodels
import brainpy as bp

# %% [markdown]
# ## Parameters

# %%
num_exc = 3200
num_inh = 800
taum = 20
taue = 5
taui = 10
Vt = -50
Vr = -60
El = -49
we = 60 * 0.27 / 10  # excitatory synaptic weight (voltage)
wi = -20 * 4.5 / 10  # inhibitory synaptic weight
ref = 5.0


# %% [markdown]
# ## Implementation 1


# %%
# neuron groups
E = brainmodels.LIF(num_exc, tau_ref=ref, V_rest=El, V_th=Vt, V_reset=Vr, tau=taum, monitors=['spike'])
I = brainmodels.LIF(num_inh, tau_ref=ref, V_rest=El, V_th=Vt, V_reset=Vr, tau=taum)
E.V[:] = Vr + bp.math.random.rand(num_exc) * (Vt - Vr)
I.V[:] = Vr + bp.math.random.rand(num_inh) * (Vt - Vr)

# %%
# synapses
E2E = brainmodels.ExpCUBA(E, E, bp.connect.FixedProb(prob=0.02), g_max=we, tau=taue)
E2I = brainmodels.ExpCUBA(E, I, bp.connect.FixedProb(prob=0.02), g_max=we, tau=taue)
I2E = brainmodels.ExpCUBA(I, E, bp.connect.FixedProb(prob=0.02), g_max=wi, tau=taui)
I2I = brainmodels.ExpCUBA(I, I, bp.connect.FixedProb(prob=0.02), g_max=wi, tau=taui)

# %%
# network
net = bp.math.jit(bp.Network(E2E, E2I, I2I, I2E, E=E, I=I))

# %%
# simulation and visualization
net.run(400., report=0.1)
bp.visualize.raster_plot(E.mon.ts, E.mon.spike, show=True)


# %% [markdown]
# ## Implementation 2

# %%
class LIF(bp.NeuGroup):
  target_backend = 'numpy'

  def __init__(self, size, **kwargs):
    super(LIF, self).__init__(size=size, **kwargs)

    self.V = bp.math.Variable(bp.math.ones(size) * Vr)
    self.Isyn = bp.math.Variable(bp.math.zeros(size))
    self.t_spike = bp.math.Variable(-1e7 * bp.math.ones(size))
    self.spike = bp.math.Variable(bp.math.zeros(size, dtype=bool))

  @bp.odeint
  def int_V(self, V, t, Isyn):
    return (Isyn + (El - V)) / taum

  def update(self, _t, _dt):
    for i in range(self.num):
      self.spike[i] = 0.
      if (_t - self.t_spike[i]) > ref:
        V = self.int_V(self.V[i], _t, self.Isyn[i])
        self.spike[i] = 0.
        if V >= Vt:
          self.V[i] = Vr
          self.spike[i] = 1.
          self.t_spike[i] = _t
        else:
          self.V[i] = V
    self.Isyn[:] = 0.


# %%
class Syn(bp.TwoEndConn):
  target_backend = 'numpy'

  def __init__(self, pre, post, conn, w, tau, **kwargs):
    super(Syn, self).__init__(pre, post, conn=conn, **kwargs)

    # parameters
    self.w = w
    self.tau = tau

    self.pre2post = self.conn.requires('pre2post')  # connections
    self.g = bp.math.Variable(bp.math.zeros(post.num))  # variables

  @bp.odeint
  def int_g(self, g, t):
    dg = - g / self.tau
    return dg

  def update(self, _t, _dt):
    self.g[:] = self.int_g(self.g, _t)
    for pre_id in range(self.pre.num):
      if self.pre.spike[pre_id] > 0.:
        post_ids = self.pre2post[pre_id]
        for i in post_ids:
          self.g[i] += self.w
    self.post.Isyn += self.g


# %%
E = LIF(num_exc, monitors=['spike'])
I = LIF(num_inh)
E.V[:] = Vr + bp.math.random.rand(num_exc) * (Vt - Vr)
I.V[:] = Vr + bp.math.random.rand(num_inh) * (Vt - Vr)

# %%
E2E = Syn(pre=E, post=E, conn=bp.connect.FixedProb(prob=0.02), w=we, tau=taue)
E2I = Syn(pre=E, post=I, conn=bp.connect.FixedProb(prob=0.02), w=we, tau=taue)
I2E = Syn(pre=I, post=E, conn=bp.connect.FixedProb(prob=0.02), w=wi, tau=taui)
I2I = Syn(pre=I, post=I, conn=bp.connect.FixedProb(prob=0.02), w=wi, tau=taui)
net = bp.Network(E, I, E2E, E2I, I2I, I2E)
net = bp.math.jit(net)

# %%
net.run(400., report=0.1)
bp.visualize.raster_plot(E.mon.ts, E.mon.spike, show=True)
