# -*- coding: utf-8 -*-
# %% [markdown]
# # _(Wang & Buzsáki, 1996)_ Gamma Oscillation

# %% [markdown]
# Here we show the implementation of gamma oscillation proposed by Xiao-Jing Wang and György Buzsáki (1996). They demonstrated that the GABA$_A$ synaptic transmission provides a suitable mechanism for synchronized gamma oscillations in a network of fast-spiking interneurons. 

# %% [markdown]
# Let's first import brainpy and set profiles.

# %%
import brainpy as bp
import brainpy.math as bm

bp.math.set_dt(0.05)


# %% [markdown]
# The network is constructed with Hodgkin–Huxley (HH) type neurons and GABA$_A$ synapses. 

# %% [markdown]
# The GABA$_A$ synapse is coded as:

# %%
class GABAa(bp.TwoEndConn):
  target_backend = 'numpy'

  def __init__(self, pre, post, conn, delay=0., g_max=0.1, E=-75.,
               alpha=12., beta=0.1, T=1.0, T_duration=1.0, method='exp_auto'):
    super(GABAa, self).__init__(pre=pre, post=post, conn=conn)
    self.check_pre_attrs('spike')
    self.check_post_attrs('t_last_spike', 'input', 'V')

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
    self.size = self.conn_mat.shape

    # variables
    self.s = bm.Variable(bm.zeros(self.size))

    # function
    ds = lambda s, t, TT: self.alpha * TT * (1 - s) - self.beta * s
    self.integral = bp.odeint(ds, method=method)

  def update(self, _t, _dt):
    TT = ((_t - self.pre.t_last_spike) < self.T_duration) * self.T
    TT = TT.reshape((-1, 1)) * self.conn_mat
    self.s.value = self.integral(self.s, _t, TT, _dt)
    self.post.input -= self.g_max * bm.sum(self.s, axis=0) * (self.post.V - self.E)


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
  def __init__(self, size, ENa=55., EK=-90., EL=-65, C=1.0, gNa=35.,
               gK=9., gL=0.1, V_th=20., phi=5.0, method='exp_auto'):
    super(HH, self).__init__(size=size)

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
    self.V = bm.Variable(bm.ones(size) * -65.)
    self.h = bm.Variable(bm.ones(size) * 0.6)
    self.n = bm.Variable(bm.ones(size) * 0.32)
    self.spike = bm.Variable(bm.zeros(size, dtype=bool))
    self.input = bm.Variable(bm.zeros(size))
    self.t_last_spike = bm.Variable(bm.ones(size) * -1e7)

    # integral
    self.integral = bp.odeint(bp.JointEq([self.dV, self.dh, self.dn]), method=method)

  def dh(self, h, t, V):
    alpha = 0.07 * bm.exp(-(V + 58) / 20)
    beta = 1 / (bm.exp(-0.1 * (V + 28)) + 1)
    dhdt = alpha * (1 - h) - beta * h
    return self.phi * dhdt

  def dn(self, n, t, V):
    alpha = -0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1)
    beta = 0.125 * bm.exp(-(V + 44) / 80)
    dndt = alpha * (1 - n) - beta * n
    return self.phi * dndt

  def dV(self, V, t, h, n, Iext):
    m_alpha = -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1)
    m_beta = 4 * bm.exp(-(V + 60) / 18)
    m = m_alpha / (m_alpha + m_beta)
    INa = self.gNa * m ** 3 * h * (V - self.ENa)
    IK = self.gK * n ** 4 * (V - self.EK)
    IL = self.gL * (V - self.EL)
    dVdt = (- INa - IK - IL + Iext) / self.C

    return dVdt

  def update(self, _t, _dt):
    V, h, n = self.integral(self.V, self.h, self.n, _t, self.input, _dt)
    self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    self.t_last_spike.value = bm.where(self.spike, _t, self.t_last_spike)
    self.V.value = V
    self.h.value = h
    self.n.value = n
    self.input[:] = 0.


# %% [markdown]
# Let's run a simulation of a network with 100 neurons with constant inputs (1 $\mu$A/cm$^2$).

# %%
num = 100
neu = HH(num)
neu.V[:] = -70. + bm.random.normal(size=num) * 20

syn = GABAa(pre=neu, post=neu, conn=bp.connect.All2All(include_self=False))
syn.g_max = 0.1 / num

net = bp.Network(neu=neu, syn=syn)
runner = bp.StructRunner(net, monitors=['neu.spike', 'neu.V'], inputs=['neu.input', 1.])
runner.run(duration=500.)

# %%
fig, gs = bp.visualize.get_figure(2, 1, 3, 8)

fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon['neu.V'], ylabel='Membrane potential (N0)')

fig.add_subplot(gs[1, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['neu.spike'], show=True)

# %% [markdown]
# We can see the result of this simulation that cells starting at random and asynchronous initial conditions quickly become synchronized and their spiking times are perfectly in-phase within 5-6 oscillatory cycles.

# %% [markdown]
# **Reference**:
#
# - Wang, Xiao-Jing, and György Buzsáki. “Gamma oscillation by synaptic inhibition in a hippocampal interneuronal network model.” Journal of neuroscience 16.20 (1996): 6402-6413.
