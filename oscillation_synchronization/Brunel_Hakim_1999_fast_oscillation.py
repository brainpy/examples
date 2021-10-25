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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # _(Brunel & Hakim, 1999)_ Fast Global Oscillation

# %% [markdown]
# Implementation of the paper:
#
# - Brunel, Nicolas, and Vincent Hakim. "Fast global oscillations in networks of integrate-and-fire neurons with low firing rates." Neural computation 11.7 (1999): 1621-1671.
#
# Author:
#
# - Chaoming Wang (chao.brain@qq.com)

# %%
import brainpy as bp

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
  target_backend = 'numpy'

  def f_v(self, V, t): return (-V + muext) / tau

  def g_v(self, V, t): return sigmaext / bp.math.sqrt(tau)

  def __init__(self, size, **kwargs):
    super(LIF, self).__init__(size, **kwargs)

    self.spike = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))
    self.not_ref = bp.math.Variable(bp.math.ones(self.num, dtype=bool))
    self.V = bp.math.Variable(bp.math.ones(self.num) * Vr)
    self.t_last_spike = bp.math.Variable(-1e7 * bp.math.ones(self.num))

    self.int_v = bp.sdeint(f=self.f_v, g=self.g_v)

  def update(self, _t, _dt):
    for i in range(self.num):
      self.spike[i] = False
      self.not_ref[i] = False
      if (_t - self.t_last_spike[i]) > taurefr:
        V = self.int_v(self.V[i], _t)
        if V > theta:
          self.spike[i] = True
          self.V[i] = Vr
          self.t_last_spike[i] = _t
        else:
          self.V[i] = V
          self.not_ref[i] = True


# %%
class Syn(bp.TwoEndConn):
  target_backend = 'numpy'

  def __init__(self, pre, post, conn, delay, **kwargs):
    super(Syn, self).__init__(pre, post, conn=conn, **kwargs)

    self.pre2post = self.conn.requires('pre2post')
    self.g = self.register_constant_delay('g', post.num, delay=delay)

  def update(self, _t, _dt):
    s = bp.math.zeros(self.post.num)
    for pre_i, spike in enumerate(self.pre.spike):
      if spike:
        for post_i in self.pre2post[pre_i]:
          s[post_i] += J
    self.g.push(s)
    self.post.V -= self.g.pull() * self.post.not_ref


# %%
group = LIF(N, monitors=['spike'])
syn = Syn(pre=group, post=group, conn=bp.connect.FixedProb(sparseness), delay=delta)
net = bp.math.jit(bp.Network(group, syn))

# %%
net.run(duration, report=0.1)
bp.visualize.raster_plot(group.mon.ts, group.mon.spike, xlim=(0, duration), show=True)
