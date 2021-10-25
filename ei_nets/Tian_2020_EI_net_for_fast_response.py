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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # *(Tian, et al., 2020)* E/I Net for fast response

# %% [markdown]
# Implementation of the paper： 
#
# - *Tian, Gengshuo, et al. "Excitation-Inhibition Balanced Neural Networks for Fast Signal Detection." Frontiers in Computational Neuroscience 14 (2020): 79.*
#
# Author:
#
# - Chaoming Wang (chao.brain@qq.com)

# %%
import brainpy as bp

# %%
# set parameters

num = 10000
num_inh = int(num * 0.2)
num_exc = num - num_inh
prob = 0.25

tau_E = 15.
tau_I = 10.
V_reset = 0.
V_threshold = 15.
f_E = 3.
f_I = 2.
mu_f = 0.1

tau_Es = 6.
tau_Is = 5.
JEE = 0.25
JEI = -1.
JIE = 0.4
JII = -1.


# %%
# define neuron type


class LIF(bp.NeuGroup):
  target_backend = 'numpy'

  def __init__(self, size, tau, **kwargs):
    super(LIF, self).__init__(size, **kwargs)

    # parameters
    self.tau = tau

    # variables
    self.V = bp.math.Variable(bp.math.zeros(size))
    self.spike = bp.math.Variable(bp.math.zeros(size, dtype=bool))
    self.input = bp.math.Variable(bp.math.zeros(size))

  @bp.odeint
  def int_V(self, V, t, Isyn):
    return (-V + Isyn) / self.tau

  def update(self, _t, _dt):
    for i in range(self.num):
      V = self.int_V(self.V[i], _t, self.input[i])
      if V >= V_threshold:
        self.spike[i] = True
        V = V_reset
      else:
        self.spike[i] = False
      self.V[i] = V
      self.input[i] = 0.


# %%
# define synapse type

class Syn(bp.TwoEndConn):
  target_backend = 'numpy'

  def __init__(self, pre, post, conn, tau, g_max, **kwargs):
    super(Syn, self).__init__(pre, post, conn=conn, **kwargs)

    # parameters
    self.tau = tau
    self.g_max = g_max

    # connections
    self.pre2post = self.conn.requires('pre2post')

    # variables
    self.s = bp.math.Variable(bp.math.zeros(post.num))

  @bp.odeint
  def ints(self, s, t):
    return - s / self.tau

  def update(self, _t, _dt):
    self.s[:] = self.ints(self.s, _t)
    for pre_i, spike in enumerate(self.pre.spike):
      if spike:
        for post_i in self.pre2post[pre_i]:
          self.s[post_i] += 1.
    self.post.input += self.g_max * self.s


# %%
# build & simulate network
E = LIF(num_exc, tau=tau_E, name='E', monitors=['spike'])
E.V[:] = bp.math.random.random(num_exc) * (V_threshold - V_reset) + V_reset

I = LIF(num_inh, tau=tau_I, name='I', monitors=['spike'])
I.V[:] = bp.math.random.random(num_inh) * (V_threshold - V_reset) + V_reset

E2I = Syn(pre=E, post=I, conn=bp.connect.FixedProb(prob=prob), tau=tau_Es, g_max=JIE)
E2E = Syn(pre=E, post=E, conn=bp.connect.FixedProb(prob=prob), tau=tau_Es, g_max=JEE)
I2I = Syn(pre=I, post=I, conn=bp.connect.FixedProb(prob=prob), tau=tau_Is, g_max=JII)
I2E = Syn(pre=I, post=E, conn=bp.connect.FixedProb(prob=prob), tau=tau_Is, g_max=JEI)

net = bp.Network(E, I, I2E, E2E, I2I, E2I)
net = bp.math.jit(net)

net.run(duration=100.,
        inputs=[('E.input', f_E * bp.math.sqrt(num) * mu_f),
                ('I.input', f_I * bp.math.sqrt(num) * mu_f)],
        report=0.1)

# %%
# visualization
fig, gs = bp.visualize.get_figure(5, 1, 1.5, 10)

fig.add_subplot(gs[:3, 0])
bp.visualize.raster_plot(E.mon.ts, E.mon.spike, xlim=(0, 100))

fig.add_subplot(gs[3:, 0])
bp.visualize.raster_plot(I.mon.ts, I.mon.spike, xlim=(0, 100), show=True)
