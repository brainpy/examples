

"""

Rewrite of liquid time-constant network from https://github.com/rtqichen/time-series-datasets
"""


import numpy as np
import os
import argparse
from _datasets import *

import brainpy as bp
import brainpy.math as bm

from enum import Enum

bm.set_environment(bm.training_mode)

if any(
    [
      not os.path.exists(p)
      for p in ['data/cheetah', 'data/gesture', 'data/har',
                'data/occupancy', 'data/ozone', 'data/person',
                'data/power', 'data/traffic']
    ]
):
  raise ValueError('Please download dataset from https://share.weiyun.com/X6L4Tdpe, '
                   'and unzip the downloaded file in the folder.')


class MappingType(Enum):
  Identity = 0
  Linear = 1
  Affine = 2


class ODESolver(Enum):
  SemiImplicit = 0
  Explicit = 1
  RungeKutta = 2


class CT_GRU(bp.DynamicalSystem):
  """CT-GRU: Continuous-time gated recurrent units.

  References
  ----------
  [1] https://arxiv.org/abs/1710.04110
  """

  def __init__(self, features_in, features_out, M=8, cell_clip=-1):
    super().__init__()

    bp.check.is_instance(self.mode, bm.TrainingMode)

    self.features_in = features_in
    self.features_out = features_out
    self.M = M
    self.cell_clip = cell_clip
    ln_tau_table = np.empty(self.M)
    tau = 1
    for i in range(self.M):
      ln_tau_table[i] = np.log(tau)
      tau = tau * (10.0 ** 0.5)
    self.ln_tau_table = bm.asarray(ln_tau_table)

    self.r_encoder = bp.layers.Dense(features_in + features_out, self.features_out * self.M)
    self.q_encoder = bp.layers.Dense(features_in + features_out, self.features_out)
    self.s_encoder = bp.layers.Dense(features_in + features_out, self.features_out * self.M)

    self.reset_state(1)

  def reset_state(self, batch_size=1):
    self.state = bm.Variable(bm.zeros((batch_size, self.features_out, self.M)), batch_axis=0)

  def update(self, *args):
    x = args[0] if len(args) == 1 else args[1]
    state = self.state.value
    h = bm.reduce_sum(state, axis=2)

    fused_input = bm.concat([x, h], axis=-1)
    ln_tau_r = bm.reshape(self.r_encoder(fused_input), [-1, self.features_out, self.M])
    sf_input_r = -bm.square(ln_tau_r - self.ln_tau_table)
    rki = bm.softmax(sf_input_r, axis=2)

    q_input = bm.reduce_sum(rki * state, axis=2)
    reset_value = bm.concat([x, q_input], axis=1)
    qk = bm.tanh(self.q_encoder(reset_value))
    qk = bm.reshape(qk, [-1, self.features_out, 1])  # in order to broadcast

    ln_tau_s = self.s_encoder(fused_input)
    ln_tau_s = bm.reshape(ln_tau_s, [-1, self.features_out, self.M])
    sf_input_s = -bm.square(ln_tau_s - self.ln_tau_table)
    ski = bm.softmax(sf_input_s, axis=2)

    h_hat_next = ((1 - ski) * state + ski * qk) * bm.exp(-1.0 / self.ln_tau_table)
    if self.cell_clip > 0:
      h_hat_next = bm.clip_by_value(h_hat_next, -self.cell_clip, self.cell_clip)
    # Compute new state
    h_next = bm.reduce_sum(h_hat_next, axis=2)
    self.state.value = h_hat_next
    return h_next


class NeuralODE(bp.DynamicalSystem):
  """Continuous-time RNN-ODE

  References
  ----------
  [1] Neural Ordinary Differential Equations
  """

  def __init__(self, features_in, features_out, cell_clip=-1):
    super().__init__()

    bp.check.is_instance(self.mode, bm.TrainingMode)
    self.features_in = features_in
    self.features_out = features_out
    self.cell_clip = cell_clip
    # Number of ODE solver steps
    self._unfolds = 6
    # Time of each ODE solver step, for variable time RNN change this
    # to a placeholder/non-trainable variable
    self._delta_t = 0.1

    self.reset_state()
    self.encoder = bp.layers.Dense(features_in + features_out, features_out)

  def reset_state(self, batch_size=1):
    self.state = bm.Variable([batch_size, self.features_out], batch_axis=0)

  def _f_prime(self, inputs, state):
    return bm.tanh(self.encoder(bm.concat([inputs, state], axis=-1)))

  def update(self, *args):
    x = args[0] if len(args) == 1 else args[1]
    # CTRNN ODE is: df/dt = NN(x) - f
    # where x is the input, and NN is a MLP.
    # Input could be:
    #    1: just the input of the RNN cell
    # or 2: input of the RNN cell merged with the current state
    state = self.state.value
    for i in range(self._unfolds):
      k1 = self._delta_t * self._f_prime(x, state)
      k2 = self._delta_t * self._f_prime(x, state + k1 * 0.5)
      k3 = self._delta_t * self._f_prime(x, state + k2 * 0.5)
      k4 = self._delta_t * self._f_prime(x, state + k3)
      state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
      if self.cell_clip > 0:
        # Optional clipping of the RNN cell to enforce stability (not needed)
        state = bm.clip_by_value(state, -self.cell_clip, self.cell_clip)
    self.state.value = state
    return state


class CT_RNN(bp.DynamicalSystem):
  """Continuous-time RNN

  References
  ----------
  [1] Latent ODEs for Irregularly-Sampled Time Series
  """

  def __init__(self, features_in, features_out, cell_clip=-1, global_feedback=False):
    super().__init__()
    self.features_in = features_in
    self.features_out = features_out
    self.global_feedback = global_feedback
    self.cell_clip = cell_clip

    # Number of ODE solver steps
    self._unfolds = 6
    # Time of each ODE solver step, for variable time RNN change this
    # to a placeholder/non-trainable variable
    self._delta_t = 0.1

    # Time-constant of the cell
    self.tau = 1

    if not self.global_feedback:
      self.encoder = bp.layers.Linear(features_in, features_out)
    else:
      self.encoder = bp.layers.Linear(features_in + features_out, features_out)
    self.reset_state()

  def reset_state(self, batch_size=1):
    self.state = bm.Variable([batch_size, self.features_out], batch_axis=0)

  def update(self, *args):
    x = args[0] if len(args) == 1 else args[1]
    # CTRNN ODE is: df/dt = NN(x) - f
    # where x is the input, and NN is a MLP.
    # Input could be: 1: just the input of the RNN cell
    # or 2: input of the RNN cell merged with the current state
    state = self.state.value
    # Input Option 1: RNNCell input
    if not self.global_feedback:
      input_f_prime = bm.tanh(self.encoder(x))
    for i in range(self._unfolds):
      # Input Option 2: RNNCell input AND RNN state
      if self.global_feedback:
        fused_input = bm.concat([x, state], axis=-1)
        input_f_prime = bm.tanh(self.encoder(fused_input))
      # df/dt
      f_prime = -state / self.tau + input_f_prime
      # If we solve this ODE with explicit euler we get
      # f(t+deltaT) = f(t) + deltaT * df/dt
      state = state + self._delta_t * f_prime
      # Optional clipping of the RNN cell to enforce stability (not needed)
      if self.cell_clip > 0:
        state = bm.clip_by_value(state, -self.cell_clip, self.cell_clip)
    self.state.value = state
    return state


class LTC(bp.DynamicalSystem):
  def __init__(self, features_in, features_out,
               input_mapping=MappingType.Affine,
               solver_type=ODESolver.SemiImplicit):
    super().__init__()

    self.features_in = features_in
    self.features_out = features_out

    # Number of ODE solver steps in one RNN step
    self._ode_solver_unfolds = 6
    self._solver_type = solver_type

    self.dt = 0.1

    self._erev_init_factor = 1.
    self._w_init_max = 1.0
    self._w_init_min = 0.01
    self._cm_init_min = 0.5
    self._cm_init_max = 0.5
    self._gleak_init_min = 1
    self._gleak_init_max = 1

    self._w_min_value = 0.00001
    self._w_max_value = 1000
    self._gleak_min_value = 0.00001
    self._gleak_max_value = 1000
    self._cm_t_min_value = 0.000001
    self._cm_t_max_value = 1000

    self._fix_cm = None
    self._fix_gleak = None
    self._fix_vleak = None

    # input mapping
    if input_mapping == MappingType.Affine:
      self.w_in = bm.TrainVar(bm.ones((features_in,)))
      self.b_in = bm.TrainVar(bm.zeros((features_in,)))
      self._map_input = lambda x: self.w_in * x + self.b_in
    elif input_mapping == MappingType.Linear:
      self.w_in = bm.TrainVar(bm.ones((features_in,)))
      self._map_input = lambda x: self.w_in * x
    elif input_mapping == MappingType.Identity:
      self._map_input = lambda x: x
    else:
      raise ValueError

    # training variables
    self.sensory_mu = bm.TrainVar(bm.random.uniform(0.3, 0.8, size=[self.features_in, self.features_out]))
    self.sensory_sigma = bm.TrainVar(bm.random.uniform(0.3, 0.8, size=[self.features_in, self.features_out]))
    self.sensory_W = bm.TrainVar(bm.random.uniform(low=self._w_init_min, high=self._w_init_max,
                                                   size=[self.features_in, self.features_out]))
    self.sensory_erev = bm.TrainVar(
      (2 * bm.random.randint(low=0, high=2, size=[self.features_in, self.features_out]) - 1) *
      self._erev_init_factor
    )

    self.mu = bm.TrainVar(bm.random.uniform(0.3, 0.8, size=[self.features_out, self.features_out]))
    self.sigma = bm.TrainVar(bm.random.uniform(0.3, 0.8, size=[self.features_out, self.features_out]))
    self.W = bm.TrainVar(bm.random.uniform(low=self._w_init_min, high=self._w_init_max,
                                           size=[self.features_out, self.features_out]))
    self.erev = bm.TrainVar(
      (2 * bm.random.randint(low=0, high=2, size=[self.features_out, self.features_out]) - 1) *
      self._erev_init_factor
    )

    if self._fix_vleak is None:
      self.vleak = bm.TrainVar(bm.random.uniform(-0.2, 0.2, size=[self.features_out, ]))
    else:
      self.vleak = bm.ones([self.features_out, ]) * self._fix_vleak

    if self._fix_gleak is None:
      if self._gleak_init_max > self._gleak_init_min:
        initializer = bm.random.uniform(self._gleak_init_min, self._gleak_init_max, self.features_out)
      else:
        initializer = bm.ones(self.features_out) * self._gleak_init_min
      self.gleak = bm.TrainVar(initializer)
    else:
      self.gleak = bm.ones(self.features_out) * self._fix_gleak

    if self._fix_cm is None:
      if self._cm_init_max > self._cm_init_min:
        initializer = bm.random.uniform(self._cm_init_min, self._cm_init_max, self.features_out)
      else:
        initializer = bm.ones(self.features_out) * self._cm_init_min
      self.cm_t = bm.TrainVar(initializer)
    else:
      self.cm_t = bm.ones(self.features_out) * self._fix_cm

    # state
    self.reset_state(1)

  def reset_state(self, batch_size=1):
    self.state = bm.Variable([batch_size, self.features_out], batch_axis=0)

  def update(self, *args):
    inputs = self._map_input(args[0] if len(args) == 1 else args[1])
    if self._solver_type == ODESolver.Explicit:
      next_state = self._ode_step_explicit(inputs, self.state.value)
    elif self._solver_type == ODESolver.SemiImplicit:
      next_state = self._ode_step(inputs, self.state.value)
    elif self._solver_type == ODESolver.RungeKutta:
      next_state = self._ode_step_runge_kutta(inputs, self.state.value)
    else:
      raise ValueError("Unknown ODE solver '{}'".format(str(self._solver_type)))
    self.state.value = next_state
    return next_state

  def _ode_step(self, inputs, state):
    # Hybrid euler method
    sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
    sensory_rev_activation = sensory_w_activation * self.sensory_erev
    w_numerator_sensory = bm.reduce_sum(sensory_rev_activation, axis=1)
    w_denominator_sensory = bm.reduce_sum(sensory_w_activation, axis=1)
    v_pre = state
    for t in range(self._ode_solver_unfolds):
      w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
      rev_activation = w_activation * self.erev
      w_numerator = bm.reduce_sum(rev_activation, axis=1) + w_numerator_sensory
      w_denominator = bm.reduce_sum(w_activation, axis=1) + w_denominator_sensory
      numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
      denominator = self.cm_t + self.gleak + w_denominator
      v_pre = numerator / denominator
    return v_pre

  def _ode_step_runge_kutta(self, inputs, state):
    for i in range(self._ode_solver_unfolds):
      k1 = self.dt * self._f_prime(inputs, state)
      k2 = self.dt * self._f_prime(inputs, state + k1 * 0.5)
      k3 = self.dt * self._f_prime(inputs, state + k2 * 0.5)
      k4 = self.dt * self._f_prime(inputs, state + k3)
      state = state + 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return state

  def _ode_step_explicit(self, inputs, state):
    # We can pre-compute the effects of the sensory neurons here
    sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
    w_reduced_sensory = bm.reduce_sum(sensory_w_activation, axis=1)
    # Unfold the ODE multiple times into one RNN step
    v_pre = state
    for t in range(self._ode_solver_unfolds):
      w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
      w_reduced_synapse = bm.reduce_sum(w_activation, axis=1)
      sensory_in = self.sensory_erev * sensory_w_activation
      synapse_in = self.erev * w_activation
      sum_in = (bm.reduce_sum(synapse_in, axis=1) - v_pre * w_reduced_synapse +
                bm.reduce_sum(sensory_in, axis=1) - v_pre * w_reduced_sensory)
      f_prime = (self.gleak * (self.vleak - v_pre) + sum_in) / self.cm_t
      v_pre = v_pre + self.dt * f_prime
    return v_pre

  def _sigmoid(self, v_pre, mu, sigma):
    v_pre = bm.reshape(v_pre, [-1, v_pre.shape[-1], 1])
    return bm.sigmoid(sigma * (v_pre - mu))

  def _f_prime(self, inputs, state):
    # We can pre-compute the effects of the sensory neurons here
    sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
    w_reduced_sensory = bm.reduce_sum(sensory_w_activation, axis=1)
    # Unfold the ODE multiple times into one RNN step
    w_activation = self.W * self._sigmoid(state, self.mu, self.sigma)
    w_reduced_synapse = bm.reduce_sum(w_activation, axis=1)
    sensory_in = self.sensory_erev * sensory_w_activation
    synapse_in = self.erev * w_activation
    sum_in = (bm.reduce_sum(sensory_in, axis=1) - state * w_reduced_synapse +
              bm.reduce_sum(synapse_in, axis=1) - state * w_reduced_sensory)
    f_prime = (self.gleak * (self.vleak - state) + sum_in) / self.cm_t
    return f_prime

  def param_constrain(self):
    if self._fix_cm is None:
      self.cm_t.value = bm.clip_by_value(self.cm_t, self._cm_t_min_value, self._cm_t_max_value)
    if self._fix_gleak is None:
      self.gleak.value = bm.clip_by_value(self.gleak, self._gleak_min_value, self._gleak_max_value)
    self.W.value = bm.clip_by_value(self.W, self._w_min_value, self._w_max_value)
    self.sensory_W.value = bm.clip_by_value(self.sensory_W, self._w_min_value, self._w_max_value)


class _BaseTask(object):
  def __init__(self,
               model_type,
               model_size,
               data,
               features_in=17,
               features_out=17,
               lr=0.001,
               ltc_lr=0.01,  # LTC needs a higher learning rate
               result_dir='results/'):
    self.model_type = model_type
    self.model_size = model_size

    # models
    if self.model_type == "lstm":
      rnn = bp.layers.LSTMCell(features_in, model_size)
    elif self.model_type.startswith("ltc"):
      lr = ltc_lr
      if self.model_type.endswith("_rk"):
        rnn = LTC(features_in, model_size, solver_type=ODESolver.RungeKutta)
      elif self.model_type.endswith("_ex"):
        rnn = LTC(features_in, model_size, solver_type=ODESolver.Explicit)
      else:
        rnn = LTC(features_in, model_size, solver_type=ODESolver.SemiImplicit)
    elif self.model_type == "node":
      rnn = NeuralODE(features_in, model_size, cell_clip=-1)
    elif self.model_type == "ctgru":
      rnn = CT_GRU(features_in, model_size, cell_clip=-1)
    elif self.model_type == "ctrnn":
      rnn = CT_RNN(features_in, model_size, cell_clip=-1, global_feedback=True)
    else:
      raise ValueError("Unknown model type '{}'".format(self.model_type))
    readout = bp.layers.Dense(model_size, features_out)
    self.model = bp.Sequential(rnn, readout)

    # optimizer
    self.optimizer = bp.optim.Adam(lr, self.model.train_vars())

    # functions
    self.f_grad = bm.grad(self._loss,
                          child_objs=self.model,
                          grad_vars=self.model.train_vars(),
                          return_value=True,
                          has_aux=True)
    self.f_train = bm.jit(self._train, child_objs=(self.f_grad, self.optimizer))
    self.f_loss = bm.jit(self._loss, child_objs=self.model)

    # data
    self.data = data

    # training metric
    self.epoch = 0
    self.best_valid_metric = 0.
    self.best_valid_stats = None

    # output dir
    self.result_dir = result_dir
    self.result_file = os.path.join(self.result_dir, "{}_{}.csv".format(self.model_type, model_size))
    if not os.path.exists(result_dir):
      os.makedirs(result_dir)

  def _loss(self, x, y):
    raise NotImplementedError

  def _train(self, xs, ys):
    grads, loss, aux = self.f_grad(xs, ys)
    self.optimizer.update(grads)
    return loss, aux

  def _metrics_at_current_epoch(self, e,
                                train_losses, train_auxs,
                                test_loss, test_aux,
                                valid_loss, valid_aux):
    raise NotImplementedError

  def _output(self, stats, extra=''):
    r = ', '.join([f"{k}={v:.5f}" for k, v in stats.items()])
    if extra:
      print(extra)
    print(r)
    with open(self.result_file, "a") as f:
      if extra:
        f.write(extra)
        f.write('\n')
      f.write(r)
      f.write('\n')

  def fit(self, epochs, verbose=True, log_period=50, batch_size=16):
    for e in range(self.epoch, self.epoch + epochs):
      train_losses, train_auxs = [], []
      for batch_x, batch_y in self.data.iterate_train(batch_size=batch_size):
        self.model.reset_state(batch_x.shape[1])
        loss, aux = self.f_train(batch_x, batch_y)
        if self.model_type.startswith('ltc'):
          self.model[0].param_constrain()
        train_losses.append(loss)
        train_auxs.append(aux)

      if verbose and (e + 1) % log_period == 0:
        self.model.reset_state(self.data.test_x.shape[1])
        test_loss, test_aux = self.f_loss(self.data.test_x, self.data.test_y)

        self.model.reset_state(self.data.valid_x.shape[1])
        valid_loss, valid_aux = self.f_loss(self.data.valid_x, self.data.valid_y)

        stats, best = self._metrics_at_current_epoch(e,
                                                     train_losses, train_auxs,
                                                     test_loss, test_aux,
                                                     valid_loss, valid_aux)
        self._output(stats)
        if best:
          self.best_valid_stats = stats
          states = {'optimizer': self.optimizer.state_dict(),
                    'model': self.model.state_dict(),
                    'epoch': e}
          bp.checkpoints.save(os.path.join(self.result_dir, self.model_type), states, e, overwrite=True)
    self._output(self.best_valid_stats, 'Best valid results:')


class _RegressionBasedTask(_BaseTask):
  def __init__(self, model_type, model_size, data, features_in, features_out, result_dir, lr=0.001):
    super().__init__(model_type, model_size, data=data, features_in=features_in,
                     features_out=features_out, lr=lr, result_dir=result_dir)

    self.best_valid_metric = np.PINF

    if not os.path.isfile(self.result_file):
      with open(self.result_file, "w") as f:
        f.write("best epoch, train loss, train mae, valid loss, valid mae, test loss, test mae\n")

  def _loss(self, x, y):
    readout = bm.for_loop(self.model, ({}, x))
    l = bm.reduce_mean(bm.square(y - readout))
    mae = bm.reduce_mean(bm.abs(y - readout))
    return l, mae

  def _metrics_at_current_epoch(self, e, train_losses, train_auxs, test_loss, test_aux, valid_loss, valid_aux):
    r = {
      'epoch': e,
      'train loss': round(float(np.mean(train_losses)), 5),
      'train mae': round(float(np.mean(train_auxs)), 5),
      'valid loss': round(float(valid_loss), 5),
      'valid mae': round(float(valid_aux), 5),
      'test loss': round(float(test_loss), 5),
      'test mae': round(float(test_aux), 5),
    }
    return r, self.best_valid_metric > valid_loss


class CheetahTask(_RegressionBasedTask):
  def __init__(self, model_type, model_size):
    super().__init__(model_type, model_size, data=CheetahData(),
                     features_in=17, features_out=17, result_dir="results/cheetah")


class PowerTask(_RegressionBasedTask):
  def __init__(self, model_type, model_size):
    super().__init__(model_type, model_size, data=PowerData(),
                     features_in=6, features_out=1, result_dir="results/power")


class TrafficTask(_RegressionBasedTask):
  def __init__(self, model_type, model_size):
    super().__init__(model_type, model_size, data=TrafficData(),
                     features_in=7, features_out=1, result_dir="results/traffic")

  def _loss(self, x, y):
    y = bm.expand_dims(y, axis=-1)
    readout = bm.for_loop(self.model, ({}, x))
    l = bm.reduce_mean(bm.square(y - readout))
    mae = bm.reduce_mean(bm.abs(y - readout))
    return l, mae


class _AccBasedTask(_BaseTask):
  def __init__(self, model_type, model_size, data, features_in, features_out, result_dir, lr=0.001, ltc_lr=0.01, ):
    super().__init__(model_type, model_size, data=data, features_in=features_in,
                     features_out=features_out, lr=lr, ltc_lr=ltc_lr, result_dir=result_dir)
    if not os.path.isfile(self.result_file):
      with open(self.result_file, "w") as f:
        f.write("best epoch, train loss, train accuracy, valid loss, valid accuracy, test loss, test accuracy\n")

  def _loss(self, x, y):
    readout = bm.for_loop(self.model, ({}, x))
    l = bp.losses.cross_entropy_loss(readout, y)
    predicts = bm.argmax(readout, axis=2)
    acc = bm.reduce_mean(bm.cast(bm.equal(predicts, y), bm.float32))
    return l, acc

  def _metrics_at_current_epoch(self, e, train_losses, train_auxs, test_loss, test_aux, valid_loss, valid_aux):
    r = {
      'epoch': e,
      'train loss': round(float(np.mean(train_losses)), 5),
      'train accuracy': round(float(np.mean(train_auxs) * 100), 5),
      'valid loss': round(float(valid_loss), 5),
      'valid accuracy': round(float(valid_aux) * 100, 5),
      'test loss': round(float(test_loss), 5),
      'test accuracy': round(float(test_aux) * 100, 5),
    }
    return r, self.best_valid_metric < valid_aux


class GestureTask(_AccBasedTask):
  def __init__(self, model_type, model_size):
    super().__init__(model_type, model_size, data=GestureData(),
                     features_in=32, features_out=5, result_dir="results/gesture")


class HarTask(_AccBasedTask):
  def __init__(self, model_type, model_size):
    super().__init__(model_type, model_size, data=HarData(),
                     features_in=561, features_out=6, result_dir="results/har")


class OccupancyTask(_AccBasedTask):
  def __init__(self, model_type, model_size):
    super().__init__(model_type, model_size, data=OccupancyData(),
                     features_in=5, features_out=2, ltc_lr=0.005,
                     result_dir="results/occupancy")


class PersonTask(_AccBasedTask):
  def __init__(self, model_type, model_size):
    super().__init__(model_type, model_size, data=PersonData(),
                     features_in=7, features_out=7, result_dir="results/person")


class SMnistTask(_AccBasedTask):
  def __init__(self, model_type, model_size):
    super().__init__(model_type, model_size, data=SMnistData(),
                     features_in=28, features_out=10, result_dir="results/smnist")

  def _loss(self, x, y):
    readout = bm.for_loop(self.model, ({}, x))[-1]
    l = bp.losses.cross_entropy_loss(readout, y)
    predicts = bm.argmax(readout, axis=1)
    acc = bm.reduce_mean(bm.cast(bm.equal(predicts, y), bm.float32))
    return l, acc


class OzoneTask(_BaseTask):
  def __init__(self, model_type, model_size, lr=0.001, ltc_lr=0.01, ):
    super().__init__(model_type, model_size, data=OzoneData(), features_in=72,
                     features_out=2, lr=lr, ltc_lr=ltc_lr, result_dir='data/ozone')

  def _loss(self, x, y):
    readout = bm.for_loop(self.model, ({}, x))
    weight = bm.cast(y, dtype=bm.float32) * 1.5 + 0.1
    l = bp.losses.cross_entropy_loss(readout, y, weight=weight)
    predicts = bm.argmax(readout, axis=2)

    lab = bm.cast(y, dtype=bm.float32)
    pred = bm.cast(predicts, dtype=bm.float32)
    # True/False positives/negatives
    tp = bm.reduce_sum(lab * pred)
    # tn = bm.reduce_sum((1 - lab) * (1 - pred))
    fp = bm.reduce_sum((1 - lab) * pred)
    fn = bm.reduce_sum(lab * (1 - pred))
    # don't divide by zero
    # Precision and Recall
    prec = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    # F1-score (Geometric mean of precision and recall)
    acc = 2 * (prec * recall) / (prec + recall + 1e-6)
    return l, (acc, prec, recall)

  def _metrics_at_current_epoch(self, e, train_losses, train_auxs, test_loss, test_aux, valid_loss, valid_aux):
    train_auxs = bm.asarray(train_auxs).to_numpy()
    test_aux = bm.asarray(test_aux).to_numpy()
    valid_aux = bm.asarray(valid_aux).to_numpy()
    r = {
      'epoch': e,
      'train loss': round(float(np.mean(train_losses)), 5),
      'train accuracy': round(float(np.mean(train_auxs[:, 0]) * 100), 5),
      'train prec': round(float(np.mean(train_auxs[:, 1]) * 100), 5),
      'train recall': round(float(np.mean(train_auxs[:, 2]) * 100), 5),
      'valid loss': round(float(valid_loss), 5),
      'valid accuracy': round(float(valid_aux[0]) * 100, 5),
      'valid prec': round(float(valid_aux[1]) * 100, 5),
      'valid recall': round(float(valid_aux[2]) * 100, 5),
      'test loss': round(float(test_loss), 5),
      'test accuracy': round(float(test_aux[0]) * 100, 5),
      'test prec': round(float(test_aux[1]) * 100, 5),
      'test recall': round(float(test_aux[2]) * 100, 5),
    }
    return r, self.best_valid_metric < valid_aux[0]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default="lstm",
                      choices=['ctrnn', 'lstm', 'ltc_rk', 'ltc_ex', 'ltc', 'node', 'ctgru'])
  parser.add_argument('--log', default=1, type=int)
  parser.add_argument('--size', default=32, type=int)
  parser.add_argument('--epochs', default=200, type=int)
  args = parser.parse_args()

  # model = CheetahModel(model_type=args.model, model_size=args.size)
  # model.fit(epochs=args.epochs, log_period=args.log)

  # model = GestureModel(model_type=args.model, model_size=args.size)
  # model.fit(epochs=args.epochs, log_period=args.log)

  # model = HarTask(model_type=args.model, model_size=args.size)
  # model.fit(epochs=args.epochs, log_period=args.log)

  # model = OccupancyTask(model_type=args.model, model_size=args.size)
  # model.fit(epochs=args.epochs, log_period=args.log)

  # model = PersonTask(model_type=args.model, model_size=args.size)
  # model.fit(epochs=args.epochs, log_period=args.log, batch_size=64)

  # model = PowerTask(model_type=args.model, model_size=args.size)
  # model.fit(epochs=args.epochs, log_period=args.log)

  # model = SMnistTask(model_type=args.model, model_size=args.size)
  # model.fit(epochs=args.epochs, log_period=args.log)

  # model = TrafficTask(model_type=args.model, model_size=args.size)
  # model.fit(epochs=args.epochs, log_period=args.log)

  model = OzoneTask(model_type=args.model, model_size=args.size)
  model.fit(epochs=args.epochs, log_period=args.log)
