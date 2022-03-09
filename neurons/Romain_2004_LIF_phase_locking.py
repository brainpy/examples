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
#     display_name: brainpy
#     language: python
#     name: brainpy
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
import brainpy.math as bm

# %%
# %matplotlib inline
import matplotlib.pyplot as plt

# %%
# set parameters
num = 2000
tau = 100.  # ms
Vth = 1.  # mV
Vr = 0.  # mV
inputs = bm.linspace(2., 4., num)


# %%
class LIF(bp.dyn.NeuGroup):
  def __init__(self, size, **kwargs):
    super(LIF, self).__init__(size, **kwargs)
    
    self.V = bm.Variable(bm.zeros(size))
    self.spike = bm.Variable(bm.zeros(size, dtype=bool))
    self.integral = bp.odeint(self.derivative)

  def derivative(self, V, t):
    return (-V + inputs + 2 * bm.sin(2 * bm.pi * t / tau)) / tau

  def update(self, _t, _dt):
    V = self.integral(self.V, _t, _dt)
    self.spike.value = V >= Vth
    self.V.value = bm.where(self.spike > 0., Vr, V)


# %%
group = LIF(num)
runner = bp.dyn.DSRunner(group, monitors=['spike'])

# %%
t = runner.run(duration=5 * 1000.)

indices, times = bp.measure.raster_plot(runner.mon.spike, runner.mon.ts)

# plt.plot((times % tau) / tau, inputs[indices], ',')

spike_phases = (times % tau) / tau
params = inputs[indices]
plt.scatter(x=spike_phases, y=params, c=spike_phases, marker=',', s=0.1, cmap="coolwarm")

plt.xlabel('Spike phase')
plt.ylabel('Parameter (input)')
plt.show()
