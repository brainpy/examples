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
# # _(Brette, Romain. 2004)_ LIF phase locking 

# %% [markdown]
# Implementation of the paper:
#
# - Brette, Romain. "Dynamics of one-dimensional spiking neuron models." Journal of mathematical biology 48.1 (2004): 38-56.
#
# Author:
#
# - Chaoming Wang (chao.brain@qq.com)

# %%
import brainpy as bp

# %%
# %matplotlib inline
import matplotlib.pyplot as plt

# %%
# set parameters
num = 2000
tau = 100.  # ms
Vth = 1.  # mV
Vr = 0.  # mV
inputs = bp.math.linspace(2., 4., num)


# %%
class LIF(bp.NeuGroup):
  def __init__(self, size, **kwargs):
    super(LIF, self).__init__(size, **kwargs)
    
    self.V = bp.math.Variable(bp.math.zeros(size))
    self.spike = bp.math.Variable(bp.math.zeros(size))

  @staticmethod
  @bp.odeint
  def int_f(V, t):
    return (-V + inputs + 2 * bp.math.sin(2 * bp.math.pi * t / tau)) / tau

  def update(self, _t, _dt):
    V = self.int_f(self.V, _t)
    self.spike[:] = bp.math.asarray(V >= Vth, dtype=bp.math.float_)
    self.V[:] = bp.math.where(self.spike > 0., Vr, V)


# %%
group = LIF(num, monitors=['spike'])
group = bp.math.jit(group)

# %%
group.run(duration=5 * 1000., report=0.1)

indices, times = bp.measure.raster_plot(group.mon.spike, group.mon.ts)

# plt.plot((times % tau) / tau, inputs[indices], ',')

spike_phases = (times % tau) / tau
params = inputs[indices]
plt.scatter(x=spike_phases, y=params, c=spike_phases, marker=',', s=0.1, cmap="coolwarm")

plt.xlabel('Spike phase')
plt.ylabel('Parameter (input)')
plt.show()
