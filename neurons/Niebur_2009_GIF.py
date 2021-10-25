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
# # *(Niebur, et. al, 2009)* Generalized integrate-and-fire model

# %% [markdown]
# Implementation of the paper: *Mihalaş, Ştefan, and Ernst Niebur. "A generalized linear integrate-and-fire neural model produces diverse spiking behaviors." Neural computation 21.3 (2009): 704-718.*

# %%
import matplotlib.pyplot as plt
import brainpy as bp
import brainmodels


# %% [markdown]
# ## Model Overview

# %% [markdown]
# Generalized integrate-and-fire model is a spiking neuron model, describes single neuron behavior and can generate most kinds of firing patterns by tuning parameters.
#
# Generalized IF model is originated from Leaky Integrate-and-Fire model (LIF model), yet it's differentiated from LIF model, for it includes internal currents $I_j$ in its expressions.

# %% [markdown]
# $$\frac{d I_j}{d t} = -k_j I_j$$
#
# $$\tau\frac{d V}{d t} = - (V - V_{rest}) + R\sum_{j}I_j + RI$$
#
# $$\frac{d V_{th}}{d t} = a(V - V_{rest}) - b(V_{th} - V_{th\infty})$$
#

# %% [markdown]
# Generalized IF neuron fire when $V$ meet $V_{th}$:

# %% [markdown]
# $$I_j \leftarrow R_j I_j + A_j$$
#
# $$V \leftarrow V_{reset}$$
#
# $$V_{th} \leftarrow max(V_{th_{reset}}, V_{th}) $$

# %% [markdown]
# ## Different firing patterns

# %% [markdown]
# These arbitrary number of internal currents $I_j$ can be seen as currents caused by ion channels' dynamics, provides the GeneralizedIF model a flexibility to generate various firing patterns.
#
# With appropriate parameters, we can reproduce most of the single neuron firing patterns. In the original paper (Mihalaş et al., 2009), the author used two internal currents $I1$ and $I2$.

# %%
def run(model, duration, I_ext):
  model.run(duration, inputs=('input', I_ext, 'iter'))

  ts = model.mon.ts
  fig, gs = bp.visualize.get_figure(1, 1, 4, 8)
  ax1 = fig.add_subplot(gs[0, 0])
  #ax1.title.set_text(f'{mode}')

  ax1.plot(ts, model.mon.V[:, 0], label='V')
  ax1.plot(ts, model.mon.V_th[:, 0], label='V_th')
  ax1.set_xlabel('Time (ms)')
  ax1.set_ylabel('Membrane potential')
  ax1.set_xlim(-0.1, ts[-1] + 0.1)
  plt.legend()

  ax2 = ax1.twinx()
  ax2.plot(ts, I_ext, color='turquoise', label='input')
  ax2.set_xlabel('Time (ms)')
  ax2.set_ylabel('External input')
  ax2.set_xlim(-0.1, ts[-1] + 0.1)
  ax2.set_ylim(-5., 20.)
  plt.legend(loc='lower left')
  plt.show()


# %% [markdown]
# Simulate Generalized IF neuron groups to generate different spiking patterns. Here we plot 20 spiking patterns in groups of 4. The plots are labeled with corresponding pattern names above the plots.

# %% [markdown]
# ### Tonic Spiking

# %%
Iext, duration = bp.inputs.constant_current([(1.5, 200.)])
neu = brainmodels.neurons.GIF(1, monitors=['V', 'V_th'])
run(neu, duration, Iext)

# %% [markdown]
# ### Class 1 Excitability

# %%
Iext, duration = bp.inputs.constant_current([(1.+1e-6, 500.)])
neu = brainmodels.neurons.GIF(1, monitors=['V', 'V_th'])
run(neu, duration, Iext)

# %% [markdown]
# ### Spike Frequency Adaptation

# %%
Iext, duration = bp.inputs.constant_current([(2., 200.)])
neu = brainmodels.neurons.GIF(1, a=0.005, monitors=['V', 'V_th'])
run(neu, duration, Iext)

# %% [markdown]
# ### Phasic Spiking

# %%
Iext, duration = bp.inputs.constant_current([(1.5, 500.)])
neu = brainmodels.neurons.GIF(1, a=0.005, monitors=['V', 'V_th'])
run(neu, duration, Iext)

# %% [markdown]
# ### Accomodation

# %%
Iext, duration = bp.inputs.constant_current([(1.5, 100.), 
                                             (0, 500.), 
                                             (0.5, 100.),
                                             (1., 100.), 
                                             (1.5, 100.), 
                                             (0., 100.)])
neu = brainmodels.neurons.GIF(1, a=0.005, monitors=['V', 'V_th'])
run(neu, duration, Iext)

# %% [markdown]
# ### Threshold Variability

# %%
Iext, duration = bp.inputs.constant_current([(1.5, 20.), 
                                             (0., 180.), 
                                             (-1.5, 20.),
                                             (0., 20.), 
                                             (1.5, 20.), 
                                             (0., 140.)])
neu = brainmodels.neurons.GIF(1, a=0.005, monitors=['V', 'V_th'])
run(neu, duration, Iext)

# %% [markdown]
# ### Rebound Spiking

# %%
Iext, duration = bp.inputs.constant_current([(0, 50.), (-3.5, 750.), (0., 200.)])
neu = brainmodels.neurons.GIF(1, a=0.005, monitors=['V', 'V_th'])
run(neu, duration, Iext)

# %% [markdown]
# ### Class 2 Excitability

# %%
Iext, duration = bp.inputs.constant_current([(2 * (1. + 1e-6), 200.)])
neu = brainmodels.neurons.GIF(1, a=0.005, monitors=['V', 'V_th'])
neu.V_th[:] = -30.
run(neu, duration, Iext)

# %% [markdown]
# ### Integrator

# %%
Iext, duration = bp.inputs.constant_current([(1.5, 20.), 
                                             (0., 10.), 
                                             (1.5, 20.), 
                                             (0., 250.),
                                             (1.5, 20.), 
                                             (0., 30.), 
                                             (1.5, 20.), 
                                             (0., 30.)])
neu = brainmodels.neurons.GIF(1, a=0.005, monitors=['V', 'V_th'])
run(neu, duration, Iext)

# %% [markdown]
# ### Input Bistability

# %%
Iext, duration = bp.inputs.constant_current([(1.5, 100.), 
                                             (1.7, 400.),
                                             (1.5, 100.), 
                                             (1.7, 400.)])
neu = brainmodels.neurons.GIF(1, a=0.005, monitors=['V', 'V_th'])
run(neu, duration, Iext)

# %% [markdown]
# ### Hyperpolarization-induced Spiking

# %%
Iext, duration = bp.inputs.constant_current([(-1., 400.)])
neu = brainmodels.neurons.GIF(1, V_th_reset=-60., V_th_inf=-120., monitors=['V', 'V_th'])
neu.V_th[:] = -50.
run(neu, duration, Iext)

# %% [markdown]
# ### Hyperpolarization-induced Bursting

# %%
Iext, duration = bp.inputs.constant_current([(-1., 400.)])
neu = brainmodels.neurons.GIF(1, V_th_reset=-60., V_th_inf=-120., A1=10., 
                              A2=-0.6, monitors=['V', 'V_th'])
neu.V_th[:] = -50.
run(neu, duration, Iext)

# %% [markdown]
# ### Tonic Bursting

# %%
Iext, duration = bp.inputs.constant_current([(2., 500.)])
neu = brainmodels.neurons.GIF(1, a=0.005, A1=10., A2=-0.6, monitors=['V', 'V_th'])
run(neu, duration, Iext)

# %% [markdown]
# ### Phasic Bursting

# %%
Iext, duration = bp.inputs.constant_current([(1.5, 500.)])
neu = brainmodels.neurons.GIF(1, a=0.005, A1=10., A2=-0.6, monitors=['V', 'V_th'])
run(neu, duration, Iext)

# %% [markdown]
# ### Rebound Bursting

# %%
Iext, duration = bp.inputs.constant_current([(0, 100.), (-3.5, 500.), (0., 400.)])
neu = brainmodels.neurons.GIF(1, a=0.005, A1=10., A2=-0.6, monitors=['V', 'V_th'])
run(neu, duration, Iext)

# %% [markdown]
# ### Mixed Mode

# %%
Iext, duration = bp.inputs.constant_current([(2., 500.)])
neu = brainmodels.neurons.GIF(1, a=0.005, A1=5., A2=-0.3, monitors=['V', 'V_th'])
run(neu, duration, Iext)

# %% [markdown]
# ### Afterpotentials

# %%
Iext, duration = bp.inputs.constant_current([(2., 15.), (0, 185.)])
neu = brainmodels.neurons.GIF(1, a=0.005, A1=5., A2=-0.3, monitors=['V', 'V_th'])
run(neu, duration, Iext)

# %% [markdown]
# ### Basal Bistability

# %%
Iext, duration = bp.inputs.constant_current([(5., 10.), (0., 90.), (5., 10.), (0., 90.)])
neu = brainmodels.neurons.GIF(1, A1=8., A2=-0.1, monitors=['V', 'V_th'])
run(neu, duration, Iext)

# %% [markdown]
# ### Preferred Frequency

# %%
Iext, duration = bp.inputs.constant_current([(5., 10.), 
                                             (0., 10.), 
                                             (4., 10.), 
                                             (0., 370.),
                                             (5., 10.), 
                                             (0., 90.), 
                                             (4., 10.), 
                                             (0., 290.)])
neu = brainmodels.neurons.GIF(1, a=0.005, A1=-3., A2=0.5, monitors=['V', 'V_th'])
run(neu, duration, Iext)

# %% [markdown]
# ### Spike Latency

# %%
Iext, duration = bp.inputs.constant_current([(8., 2.), (0, 48.)])
neu = brainmodels.neurons.GIF(1, a=-0.08, monitors=['V', 'V_th'])
run(neu, duration, Iext)
