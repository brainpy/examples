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
# # _(Bouchacourt & Buschman, 2019)_ Flexible Working Memory Model

# %% [markdown]
# Implementation of :
#
# - Bouchacourt, Flora, and Timothy J. Buschman. "A flexible model of working memory." Neuron 103.1 (2019): 147-160.
#
# Author:
#
# - Chaoming Wang (chao.brain@qq.com)

# %%
import matplotlib.pyplot as plt
import numba

# %%
import brainpy as bp

dt = bp.math.get_dt()

# %%
# increase in order to run multiple trials with the same network
num_trials = 1
num_item_to_load = 6

# %%
# Parameters for network architecture
# ------------------------------------

num_sensory_neuron = 512  # Number of recurrent neurons per sensory network
num_sensory_pool = 8  # Number of ring-like sensory network
num_all_sensory = num_sensory_pool * num_sensory_neuron
num_all_random = 1024  # Number of neuron in the random network
fI_slope = 0.4  # slope of the non-linear f-I function
bias = 0.  # bias in the neuron firing response (cf page 1 right column of Burak, Fiete 2012)
tau = 10.  # Synaptic time constant [ms]
init_range = 0.01  # Range to randomly initialize synaptic variables

# %%
# Parameters for sensory network
# -------------------------------

k2 = 0.25  # width of negative surround suppression
k1 = 1.  # width of positive amplification
A = 2.  # amplitude of weight function
lambda_ = 0.28  # baseline of weight matrix for recurrent network

# %%
# Parameters for interaction of
# random network <-> sensory network
# -----------------------------------

forward_balance = -1.  # if -1, perfect feed-forward balance from SN to RN
backward_balance = -1.  # if -1, perfect feedback balance from RN to SN
alpha = 2.1  # parameter used to compute the feedforward weight, before balancing
beta = 0.2  # parameter used to compute the feedback weight, before balancing
gamma = 0.35  # connectivity (gamma in the paper)
factor = 1000  # factor for computing weights values

# %%
# Parameters for stimulus
# -----------------------

simulation_time = 1100  # # the simulation time [ms]
start_stimulation = 100  # [ms]
end_stimulation = 200  # [ms]
input_strength = 10  # strength of the stimulation
num_sensory_input_width = 32
# the width for input stimulation of the gaussian distribution
sigma = round(num_sensory_neuron / num_sensory_input_width)
three_sigma = 3 * sigma
activity_threshold = 3

# %%
# Weights initialization
# ----------------------

# weight matrix within sensory network
sensory_encoding = 2. * bp.math.pi * bp.math.arange(1, num_sensory_neuron + 1) / num_sensory_neuron
diff = sensory_encoding.reshape((-1, 1)) - sensory_encoding
weight_mat_of_sensory = lambda_ + A * bp.math.exp(k1 * (bp.math.cos(diff) - 1)) - \
                        A * bp.math.exp(k2 * (bp.math.cos(diff) - 1))
diag = bp.math.arange(num_sensory_neuron)
weight_mat_of_sensory[diag, diag] = 0.

# %%
# connectivity matrix between sensory and random network
conn_matrix_sensory2random = bp.math.random.rand(num_all_sensory, num_all_random) < gamma

# %%
# weight matrix of sensory2random
ws = factor * alpha / conn_matrix_sensory2random.sum(axis=0)
weight_mat_sensory2random = conn_matrix_sensory2random * ws.reshape((1, -1))
ws = weight_mat_sensory2random.sum(axis=0).reshape((1, -1))
weight_mat_sensory2random += forward_balance / num_all_sensory * ws  # balance

# %%
# weight matrix of random2sensory
ws = factor * beta / conn_matrix_sensory2random.sum(axis=1)
weight_mat_random2sensory = conn_matrix_sensory2random.T * ws.reshape((1, -1))
ws = weight_mat_random2sensory.sum(axis=0).reshape((1, -1))
weight_mat_random2sensory += backward_balance / num_all_random * ws  # balance


# %%
@numba.njit(['float64[:](float64)'])
def get_input(input_center):
  input = bp.math.zeros(num_sensory_neuron)
  inp_ids = bp.math.arange(int(round(input_center - three_sigma)),
                           int(round(input_center + three_sigma + 1)), 1)
  inp_scale = bp.math.exp(-(inp_ids - input_center) ** 2 / 2 / sigma ** 2) / (bp.math.sqrt(2 * bp.math.pi) * sigma)
  inp_scale /= bp.math.max(inp_scale)
  inp_ids = bp.math.remainder(inp_ids - 1, num_sensory_neuron)
  input[inp_ids] = input_strength * inp_scale
  input -= bp.math.sum(input) / input.shape[0]
  return input


# %%
def get_activity_vector(rates):
  exp_stim_encoding = bp.math.exp(1j * sensory_encoding, dtype=complex)
  timed_abs = bp.math.zeros(num_sensory_pool)
  timed_angle = bp.math.zeros(num_sensory_pool)
  for si in range(num_sensory_pool):
    start = si * num_sensory_neuron
    end = (si + 1) * num_sensory_neuron
    exp_rates = bp.math.multiply(rates[start:end], exp_stim_encoding, dtype=complex)
    mean_rates = bp.math.mean(exp_rates)
    timed_angle[si] = bp.math.angle(mean_rates) * num_sensory_neuron / (2 * bp.math.pi)
    timed_abs[si] = bp.math.absolute(mean_rates)
  timed_angle[timed_angle < 0] += num_sensory_neuron
  return timed_abs, timed_angle


# %%
class PoissonNeuron(bp.NeuGroup):
  target_backend = 'numpy'

  @bp.odeint(method='exponential_euler')
  def int_s(self, s, t):
    ds = -s / tau
    return ds

  def __init__(self, size, **kwargs):
    super(PoissonNeuron, self).__init__(size=size, **kwargs)

    self.s = bp.math.Variable(bp.math.zeros(self.num))
    self.r = bp.math.Variable(bp.math.zeros(self.num))
    self.input = bp.math.Variable(bp.math.zeros(self.num))
    self.spike = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))

  def update(self, _t, _dt):
    self.s[:] = self.int_s(self.s, _t)
    self.r[:] = 0.4 * (1. + bp.math.tanh(fI_slope * (self.input + self.s + bias) - 3.)) / tau
    self.spike[:] = bp.math.random.random(self.s.shape) < self.r * dt
    self.input[:] = 0.

  def reset(self):
    self.s[:] = bp.math.random.random(self.num) * init_range
    self.r[:] = 0.4 * (1. + bp.math.tanh(fI_slope * (bias + self.s) - 3.)) / tau
    self.input[:] = bp.math.zeros(self.num)
    self.spike[:] = bp.math.zeros(self.num, dtype=bool)


# %%
sensory_net = PoissonNeuron(num_all_sensory, monitors=['r', 'spike'], name='S')
random_net = PoissonNeuron(num_all_random, monitors=['spike'], name='R')


# %%
class Sen2SenSyn(bp.TwoEndConn):
  target_backend = 'numpy'

  def __init__(self, pre, post, **kwargs):
    super(Sen2SenSyn, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t, _dt):
    for i in range(num_sensory_pool):
      start = i * num_sensory_neuron
      end = (i + 1) * num_sensory_neuron
      for j in range(num_sensory_neuron):
        if self.pre.spike[start + j]:
          self.post.s[start: end] += weight_mat_of_sensory[j]


sensory2sensory_conn = Sen2SenSyn(pre=sensory_net, post=sensory_net)


# %%
class OtherSyn(bp.TwoEndConn):
  target_backend = 'numpy'

  def __init__(self, pre, post, weights, **kwargs):
    self.weights = weights
    super(OtherSyn, self).__init__(pre=pre, post=post, **kwargs)

  def update(self, _t, _dt):
    for i in range(self.pre.spike.shape[0]):
      if self.pre.spike[i]:
        self.post.s += self.weights[i]


random2sensory_conn = OtherSyn(pre=random_net,
                               post=sensory_net,
                               weights=weight_mat_random2sensory)

sensory2random_conn = OtherSyn(pre=sensory_net,
                               post=random_net,
                               weights=weight_mat_sensory2random)


# %%
net = bp.Network(sensory_net, random_net, sensory2sensory_conn,
                 random2sensory_conn, sensory2random_conn)
net = bp.math.jit(net)

# %%
for trial_idx in range(num_trials):
  # inputs
  # ------
  pools_receiving_inputs = bp.math.random.choice(num_sensory_pool, num_item_to_load, replace=False)
  print(f"Load {num_item_to_load} items in trial {trial_idx}.\n")

  input_center = bp.math.ones(num_sensory_pool) * num_sensory_neuron / 2
  inp_vector = bp.math.zeros((num_sensory_pool, num_sensory_neuron))
  for si in pools_receiving_inputs:
    inp_vector[si, :] = get_input(input_center[si])
  inp_vector = inp_vector.flatten()
  Iext, duration = bp.inputs.constant_current([(0., start_stimulation),
                                               (inp_vector, end_stimulation - start_stimulation),
                                               (0., simulation_time - end_stimulation)])

  # running
  # -------
  sensory_net.reset()
  random_net.reset()

  net.run(duration, inputs=('S.input', Iext, 'iter'))

  # results
  # --------

  rate_abs, rate_angle = get_activity_vector(sensory_net.mon.r[-1] * 1e3)
  print(f"Stimulus is given in: {bp.math.sort(pools_receiving_inputs)}")
  print(f"Memory is found in: {bp.math.where(rate_abs > activity_threshold)[0]}")

  prob_maintained, prob_spurious = 0, 0
  for si in range(num_sensory_pool):
    if rate_abs[si] > activity_threshold:
      if si in pools_receiving_inputs:
        prob_maintained += 1
      else:
        prob_spurious += 1
  print(str(prob_maintained) + ' maintained memories')
  print(str(pools_receiving_inputs.shape[0] - prob_maintained) + ' forgotten memories')
  print(str(prob_spurious) + ' spurious memories\n')
  prob_maintained /= float(num_item_to_load)
  if num_item_to_load != num_sensory_pool:
    prob_spurious /= float(num_sensory_pool - num_item_to_load)

  # visualization
  # -------------
  fig, gs = bp.visualize.get_figure(6, 1, 1.5, 12)
  xlim = (0, duration)

  fig.add_subplot(gs[0:4, 0])
  bp.visualize.raster_plot(sensory_net.mon.ts, sensory_net.mon.spike, ylabel='Sensory Network', xlim=xlim)
  for index_sn in range(num_sensory_pool + 1):
    plt.axhline(index_sn * num_sensory_neuron)
  plt.yticks([num_sensory_neuron * (i + 0.5) for i in range(num_sensory_pool)],
             [f'pool-{i}' for i in range(num_sensory_pool)])

  fig.add_subplot(gs[4:6, 0])
  bp.visualize.raster_plot(sensory_net.mon.ts, random_net.mon.spike, ylabel='Random Network', xlim=xlim, show=True)
