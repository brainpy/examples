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
# # _(Wang & Buzsáki, 1996)_ Gamma Oscillation

# %% [markdown]
# Here we show the implementation of gamma oscillation proposed by Xiao-Jing Wang and György Buzsáki (1996). They demonstrated that the GABA$_A$ synaptic transmission provides a suitable mechanism for synchronized gamma oscillations in a network of fast-spiking interneurons. 

# %% [markdown]
# Let's first import brainpy and set profiles.

# %%
import brainpy as bp

bp.integrators.set_default_odeint('rk4')


# %% [markdown]
# The network is constructed with Hodgkin–Huxley (HH) type neurons and GABA$_A$ synapses. 

# %% [markdown]
# The GABA$_A$ synapse is coded as:

# %%
class GABAa(bp.TwoEndConn):
  target_backend = 'numpy'

  def __init__(self, pre, post, conn, delay=0., g_max=0.1, E=-75.,
               alpha=12., beta=0.1, T=1.0, T_duration=1.0, **kwargs):
    super(GABAa, self).__init__(pre=pre, post=post, conn=conn, **kwargs)

    # parameters
    self.g_max = g_max
    self.E = E
    self.alpha = alpha
    self.beta = beta
    self.T = T
    self.T_duration = T_duration
    self.delay = delay

    # connections
    self.conn_mat = self.conn.requires('conn_mat')
    self.size = bp.math.shape(self.conn_mat)

    # variables
    self.s = bp.math.Variable(bp.math.zeros(self.size))
    self.t_last_pre_spike = bp.math.Variable(bp.math.ones(self.size) * -1e7)
    self.g = self.register_constant_delay('g', size=self.size, delay=delay)

  @bp.odeint
  def int_s(self, s, t, TT):
    return self.alpha * TT * (1 - s) - self.beta * s

  def update(self, _t, _dt):
    for i in range(self.pre.size[0]):
      if self.pre.spike[i] > 0:
        self.t_last_pre_spike[i] = _t
    TT = ((_t - self.t_last_pre_spike) < self.T_duration) * self.T
    self.s[:] = self.int_s(self.s, _t, TT)
    self.g.push(self.g_max * self.s)
    g = self.g.pull()
    self.post.input[:] -= bp.math.sum(g, axis=0) * (self.post.V - self.E)


# %% [markdown]
# The dynamics of the HH type neurons is given by:
#
# $$ C \frac {dV} {dt} = -(I_{Na} + I_{K} + I_L) + I(t) $$
#
# where $I(t)$ is the injected current, the leak current $ I_L = g_L (V - E_L) $, and the transient sodium current 
#
# $$ I_{Na} = g_{Na} m_{\infty}^3 h (V - E_{Na}) $$
#
# where the activation variable $m$ is assumed fast and substituted by its steady-state function 
# $m_{\infty} = \alpha_m / (\alpha_m + \beta_m)$.
# And the inactivation variable $h$ obeys a first=order kinetics:
# $$ \frac {dh} {dt} = \phi (\alpha_h (1-h) - \beta_h h)$$
#
# $$ I_K = g_K n^4 (V - E_K) $$
#
# where the activation variable $n$ also obeys a first=order kinetics:
# $$ \frac {dn} {dt} = \phi (\alpha_n (1-n) - \beta_n n)$$

# %%
class HH(bp.NeuGroup):
  def __init__(self, size, ENa=55., EK=-90., EL=-65, C=1.0,
               gNa=35., gK=9., gL=0.1, V_th=20., phi=5.0, **kwargs):
    super(HH, self).__init__(size=size, **kwargs)

    # parameters
    self.ENa = ENa
    self.EK = EK
    self.EL = EL
    self.C = C
    self.gNa = gNa
    self.gK = gK
    self.gL = gL
    self.V_th = V_th
    self.phi = phi

    # variables
    self.V = bp.math.Variable(bp.math.ones(size) * -65.)
    self.h = bp.math.Variable(bp.math.ones(size) * 0.6)
    self.n = bp.math.Variable(bp.math.ones(size) * 0.32)
    self.spike = bp.math.Variable(bp.math.zeros(size, dtype=bool))
    self.input = bp.math.Variable(bp.math.zeros(size))

  @bp.odeint
  def integral(self, V, h, n, t, Iext):
    alpha = 0.07 * bp.math.exp(-(V + 58) / 20)
    beta = 1 / (bp.math.exp(-0.1 * (V + 28)) + 1)
    dhdt = alpha * (1 - h) - beta * h

    alpha = -0.01 * (V + 34) / (bp.math.exp(-0.1 * (V + 34)) - 1)
    beta = 0.125 * bp.math.exp(-(V + 44) / 80)
    dndt = alpha * (1 - n) - beta * n

    m_alpha = -0.1 * (V + 35) / (bp.math.exp(-0.1 * (V + 35)) - 1)
    m_beta = 4 * bp.math.exp(-(V + 60) / 18)
    m = m_alpha / (m_alpha + m_beta)
    INa = self.gNa * m ** 3 * h * (V - self.ENa)
    IK = self.gK * n ** 4 * (V - self.EK)
    IL = self.gL * (V - self.EL)
    dVdt = (- INa - IK - IL + Iext) / self.C

    return dVdt, self.phi * dhdt, self.phi * dndt

  def update(self, _t, _dt):
    V, h, n = self.integral(self.V, self.h, self.n, _t, self.input)
    self.spike[:] = bp.math.logical_and(self.V < self.V_th, V >= self.V_th)
    self.V[:] = V
    self.h[:] = h
    self.n[:] = n
    self.input[:] = 0.


# %% [markdown]
# Let's run a simulation of a network with 100 neurons with constant inputs (1 $\mu$A/cm$^2$).

# %%
num = 100
neu = HH(num, monitors=['spike', 'V'], name='X')
neu.V[:] = -70. + bp.math.random.normal(size=num) * 20

syn = GABAa(pre=neu, post=neu, conn=bp.connect.All2All(include_self=False))
syn.g_max = 0.1 / num

net = bp.math.jit(bp.Network(neu, syn))
net.run(duration=500., inputs=['X.input', 1.], report=0.1)

fig, gs = bp.visualize.get_figure(2, 1, 3, 8)

fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(neu.mon.ts, neu.mon.V, ylabel='Membrane potential (N0)')

fig.add_subplot(gs[1, 0])
bp.visualize.raster_plot(neu.mon.ts, neu.mon.spike, show=True)

# %% [markdown]
# We can see the result of this simulation that cells starting at random and asynchronous initial conditions quickly become synchronized and their spiking times are perfectly in-phase within 5-6 oscillatory cycles.

# %% [markdown]
# **Reference**:
#
# - Wang, Xiao-Jing, and György Buzsáki. “Gamma oscillation by synaptic inhibition in a hippocampal interneuronal network model.” Journal of neuroscience 16.20 (1996): 6402-6413.

# %% [markdown]
#
# **Author**:
#
# - Chaoming Wang (chao.brain@qq.com)
