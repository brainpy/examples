# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt

import brainpy as bp
import brainpy.math as bm


# bm.set_platform('cpu')


class PoissonStim(bp.NeuGroup):
  def __init__(self, size, freq_mean, freq_var, t_interval):
    super(PoissonStim, self).__init__(size=size)

    # parameters
    self.freq_mean = freq_mean
    self.freq_var = freq_var
    self.t_interval = t_interval
    self.dt = bm.get_dt() / 1000.

    # variables
    self.freq = bp.init.variable_(bm.zeros, 1, self.mode)
    self.freq_t_last_change = bp.init.variable_(lambda s: bm.ones(s) * -1e7, 1, self.mode)
    self.spike = bp.init.variable_(lambda s: bm.zeros(s, dtype=bool), self.varshape, self.mode)
    self.rng = bm.random.RandomState()

  def reset_state(self, batch_size=None):
    self.freq.value = bp.init.variable_(bm.zeros, 1, batch_size)
    self.freq_t_last_change.value = bp.init.variable_(lambda s: bm.ones(s) * -1e7, 1, batch_size)
    self.spike.value = bp.init.variable_(lambda s: bm.zeros(s, dtype=bool), self.varshape, batch_size)

  def update(self, tdi):
    t, dt = tdi['t'], tdi['dt']
    in_interval = bm.logical_and(pre_stimulus_period < t, t < pre_stimulus_period + stimulus_period)
    in_interval = bm.ones_like(self.freq, dtype=bool) * in_interval
    prev_freq = bm.where(in_interval, self.freq, 0.)
    in_interval = bm.logical_and(in_interval, (t - self.freq_t_last_change) >= self.t_interval)
    self.freq.value = bm.where(in_interval, self.rng.normal(self.freq_mean, self.freq_var, self.freq.shape), prev_freq)
    self.freq_t_last_change.value = bm.where(in_interval, t, self.freq_t_last_change)
    shape = (self.spike.shape[:1] + self.varshape) if isinstance(self.mode, bm.BatchingMode) else self.varshape
    self.spike.value = self.rng.random(shape) < self.freq * self.dt


class DecisionMaking(bp.Network):
  def __init__(self, scale=1., mu0=40., coherence=25.6, f=0.15):
    super(DecisionMaking, self).__init__()

    num_exc = int(1600 * scale)
    num_inh = int(400 * scale)
    num_A = int(f * num_exc)
    num_B = int(f * num_exc)
    num_N = num_exc - num_A - num_B
    print(f'Total network size: {num_exc + num_inh}')

    poisson_freq = 2400.  # Hz
    w_pos = 1.7
    w_neg = 1. - f * (w_pos - 1.) / (1. - f)
    g_ext2E_AMPA = 2.1  # nS
    g_ext2I_AMPA = 1.62  # nS
    g_E2E_AMPA = 0.05 / scale  # nS
    g_E2I_AMPA = 0.04 / scale  # nS
    g_E2E_NMDA = 0.165 / scale  # nS
    g_E2I_NMDA = 0.13 / scale  # nS
    g_I2E_GABAa = 1.3 / scale  # nS
    g_I2I_GABAa = 1.0 / scale  # nS

    ampa_par = dict(delay_step=int(0.5 / bm.get_dt()), tau=2.0)
    gaba_par = dict(delay_step=int(0.5 / bm.get_dt()), tau=5.0)
    nmda_par = dict(delay_step=int(0.5 / bm.get_dt()), tau_decay=100, tau_rise=2., a=0.5)

    # E neurons/pyramid neurons
    A = bp.neurons.LIF(num_A, V_rest=-70., V_reset=-55., V_th=-50., tau=20., R=0.04,
                       tau_ref=2., V_initializer=bp.init.OneInit(-70.))
    B = bp.neurons.LIF(num_B, V_rest=-70., V_reset=-55., V_th=-50., tau=20., R=0.04,
                       tau_ref=2., V_initializer=bp.init.OneInit(-70.))
    N = bp.neurons.LIF(num_N, V_rest=-70., V_reset=-55., V_th=-50., tau=20., R=0.04,
                       tau_ref=2., V_initializer=bp.init.OneInit(-70.))
    # I neurons/interneurons
    I = bp.neurons.LIF(num_inh, V_rest=-70., V_reset=-55., V_th=-50., tau=10., R=0.05,
                       tau_ref=1., V_initializer=bp.init.OneInit(-70.))

    # poisson stimulus
    IA = PoissonStim(num_A, freq_var=10., t_interval=50., freq_mean=mu0 + mu0 / 100. * coherence)
    IB = PoissonStim(num_B, freq_var=10., t_interval=50., freq_mean=mu0 - mu0 / 100. * coherence)

    # noise neurons
    self.noise_B = bp.neurons.PoissonGroup(num_B, freqs=poisson_freq)
    self.noise_A = bp.neurons.PoissonGroup(num_A, freqs=poisson_freq)
    self.noise_N = bp.neurons.PoissonGroup(num_N, freqs=poisson_freq)
    self.noise_I = bp.neurons.PoissonGroup(num_inh, freqs=poisson_freq)

    # define external inputs
    self.IA2A = bp.synapses.Exponential(IA, A, bp.conn.One2One(), g_max=g_ext2E_AMPA,
                                        output=bp.synouts.COBA(E=0.), **ampa_par)
    self.IB2B = bp.synapses.Exponential(IB, B, bp.conn.One2One(), g_max=g_ext2E_AMPA,
                                        output=bp.synouts.COBA(E=0.), **ampa_par)

    # define E->E/I conn

    self.N2B_AMPA = bp.synapses.Exponential(N, B, bp.conn.All2All(), g_max=g_E2E_AMPA * w_neg,
                                            output=bp.synouts.COBA(E=0.), **ampa_par)
    self.N2A_AMPA = bp.synapses.Exponential(N, A, bp.conn.All2All(), g_max=g_E2E_AMPA * w_neg,
                                            output=bp.synouts.COBA(E=0.), **ampa_par)
    self.N2N_AMPA = bp.synapses.Exponential(N, N, bp.conn.All2All(), g_max=g_E2E_AMPA,
                                            output=bp.synouts.COBA(E=0.), **ampa_par)
    self.N2I_AMPA = bp.synapses.Exponential(N, I, bp.conn.All2All(), g_max=g_E2I_AMPA,
                                            output=bp.synouts.COBA(E=0.), **ampa_par)
    self.N2B_NMDA = bp.synapses.NMDA(N, B, bp.conn.All2All(), g_max=g_E2E_NMDA * w_neg,
                                     output=bp.synouts.MgBlock(E=0., cc_Mg=1.), **nmda_par)
    self.N2A_NMDA = bp.synapses.NMDA(N, A, bp.conn.All2All(), g_max=g_E2E_NMDA * w_neg,
                                     output=bp.synouts.MgBlock(E=0., cc_Mg=1.), **nmda_par)
    self.N2N_NMDA = bp.synapses.NMDA(N, N, bp.conn.All2All(), g_max=g_E2E_NMDA,
                                     output=bp.synouts.MgBlock(E=0., cc_Mg=1.), **nmda_par)
    self.N2I_NMDA = bp.synapses.NMDA(N, I, bp.conn.All2All(), g_max=g_E2I_NMDA,
                                     output=bp.synouts.MgBlock(E=0., cc_Mg=1.), **nmda_par)

    self.B2B_AMPA = bp.synapses.Exponential(B, B, bp.conn.All2All(), g_max=g_E2E_AMPA * w_pos,
                                            output=bp.synouts.COBA(E=0.), **ampa_par)
    self.B2A_AMPA = bp.synapses.Exponential(B, A, bp.conn.All2All(), g_max=g_E2E_AMPA * w_neg,
                                            output=bp.synouts.COBA(E=0.), **ampa_par)
    self.B2N_AMPA = bp.synapses.Exponential(B, N, bp.conn.All2All(), g_max=g_E2E_AMPA,
                                            output=bp.synouts.COBA(E=0.), **ampa_par)
    self.B2I_AMPA = bp.synapses.Exponential(B, I, bp.conn.All2All(), g_max=g_E2I_AMPA,
                                            output=bp.synouts.COBA(E=0.), **ampa_par)
    self.B2B_NMDA = bp.synapses.NMDA(B, B, bp.conn.All2All(), g_max=g_E2E_NMDA * w_pos,
                                     output=bp.synouts.MgBlock(E=0., cc_Mg=1.), **nmda_par)
    self.B2A_NMDA = bp.synapses.NMDA(B, A, bp.conn.All2All(), g_max=g_E2E_NMDA * w_neg,
                                     output=bp.synouts.MgBlock(E=0., cc_Mg=1.), **nmda_par)
    self.B2N_NMDA = bp.synapses.NMDA(B, N, bp.conn.All2All(), g_max=g_E2E_NMDA,
                                     output=bp.synouts.MgBlock(E=0., cc_Mg=1.), **nmda_par)
    self.B2I_NMDA = bp.synapses.NMDA(B, I, bp.conn.All2All(), g_max=g_E2I_NMDA,
                                     output=bp.synouts.MgBlock(E=0., cc_Mg=1.), **nmda_par)

    self.A2B_AMPA = bp.synapses.Exponential(A, B, bp.conn.All2All(), g_max=g_E2E_AMPA * w_neg,
                                            output=bp.synouts.COBA(E=0.), **ampa_par)
    self.A2A_AMPA = bp.synapses.Exponential(A, A, bp.conn.All2All(), g_max=g_E2E_AMPA * w_pos,
                                            output=bp.synouts.COBA(E=0.), **ampa_par)
    self.A2N_AMPA = bp.synapses.Exponential(A, N, bp.conn.All2All(), g_max=g_E2E_AMPA,
                                            output=bp.synouts.COBA(E=0.), **ampa_par)
    self.A2I_AMPA = bp.synapses.Exponential(A, I, bp.conn.All2All(), g_max=g_E2I_AMPA,
                                            output=bp.synouts.COBA(E=0.), **ampa_par)
    self.A2B_NMDA = bp.synapses.NMDA(A, B, bp.conn.All2All(), g_max=g_E2E_NMDA * w_neg,
                                     output=bp.synouts.MgBlock(E=0., cc_Mg=1.), **nmda_par)
    self.A2A_NMDA = bp.synapses.NMDA(A, A, bp.conn.All2All(), g_max=g_E2E_NMDA * w_pos,
                                     output=bp.synouts.MgBlock(E=0., cc_Mg=1.), **nmda_par)
    self.A2N_NMDA = bp.synapses.NMDA(A, N, bp.conn.All2All(), g_max=g_E2E_NMDA,
                                     output=bp.synouts.MgBlock(E=0., cc_Mg=1.), **nmda_par)
    self.A2I_NMDA = bp.synapses.NMDA(A, I, bp.conn.All2All(), g_max=g_E2I_NMDA,
                                     output=bp.synouts.MgBlock(E=0., cc_Mg=1.), **nmda_par)

    # define I->E/I conn
    self.I2B = bp.synapses.Exponential(I, B, bp.conn.All2All(), g_max=g_I2E_GABAa,
                                       output=bp.synouts.COBA(E=-70.), **gaba_par)
    self.I2A = bp.synapses.Exponential(I, A, bp.conn.All2All(), g_max=g_I2E_GABAa,
                                       output=bp.synouts.COBA(E=-70.), **gaba_par)
    self.I2N = bp.synapses.Exponential(I, N, bp.conn.All2All(), g_max=g_I2E_GABAa,
                                       output=bp.synouts.COBA(E=-70.), **gaba_par)
    self.I2I = bp.synapses.Exponential(I, I, bp.conn.All2All(), g_max=g_I2I_GABAa,
                                       output=bp.synouts.COBA(E=-70.), **gaba_par)

    # define external projections
    self.noise2B = bp.synapses.Exponential(self.noise_B, B, bp.conn.One2One(), g_max=g_ext2E_AMPA,
                                           output=bp.synouts.COBA(E=0.), **ampa_par)
    self.noise2A = bp.synapses.Exponential(self.noise_A, A, bp.conn.One2One(), g_max=g_ext2E_AMPA,
                                           output=bp.synouts.COBA(E=0.), **ampa_par)
    self.noise2N = bp.synapses.Exponential(self.noise_N, N, bp.conn.One2One(), g_max=g_ext2E_AMPA,
                                           output=bp.synouts.COBA(E=0.), **ampa_par)
    self.noise2I = bp.synapses.Exponential(self.noise_I, I, bp.conn.One2One(), g_max=g_ext2I_AMPA,
                                           output=bp.synouts.COBA(E=0.), **ampa_par)

    # nodes
    self.B = B
    self.A = A
    self.N = N
    self.I = I
    self.IA = IA
    self.IB = IB


def visualize_raster(ax, mon, t_start=0., title=None):
  bp.visualize.raster_plot(mon['ts'], mon['A.spike'], markersize=1, ax=ax, color='', label="Group A")
  bp.visualize.raster_plot(mon['ts'], mon['B.spike'], markersize=1, ax=ax, color='', label="Group B")
  if title:
    ax.set_title(title)
  ax.set_ylabel("Neuron Index")
  ax.set_xlim(t_start, total_period + 1)
  ax.axvline(pre_stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period + delay_period, linestyle='dashed')
  ax.legend()


def visualize_results(axes, mon, t_start=0., title=None):
  ax = axes[0]
  bp.visualize.raster_plot(mon['ts'], mon['A.spike'], markersize=1, ax=ax)
  if title:
    ax.set_title(title)
  ax.set_ylabel("Group A")
  ax.set_xlim(t_start, total_period + 1)
  ax.axvline(pre_stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period + delay_period, linestyle='dashed')

  ax = axes[1]
  bp.visualize.raster_plot(mon['ts'], mon['B.spike'], markersize=1, ax=ax)
  ax.set_ylabel("Group B")
  ax.set_xlim(t_start, total_period + 1)
  ax.axvline(pre_stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period + delay_period, linestyle='dashed')

  ax = axes[2]
  rateA = bp.measure.firing_rate(mon['A.spike'], width=10.)
  rateB = bp.measure.firing_rate(mon['B.spike'], width=10.)
  ax.plot(mon['ts'], rateA, label="Group A")
  ax.plot(mon['ts'], rateB, label="Group B")
  ax.set_ylabel('Population activity [Hz]')
  ax.set_xlim(t_start, total_period + 1)
  ax.axvline(pre_stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period + delay_period, linestyle='dashed')
  ax.legend()

  ax = axes[3]
  ax.plot(mon['ts'], mon['IA.freq'], label="group A")
  ax.plot(mon['ts'], mon['IB.freq'], label="group B")
  ax.set_ylabel("Input activity [Hz]")
  ax.set_xlim(t_start, total_period + 1)
  ax.axvline(pre_stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period + delay_period, linestyle='dashed')
  ax.legend()
  ax.set_xlabel("Time [ms]")


pre_stimulus_period = 100.
stimulus_period = 1000.
delay_period = 500.
total_period = pre_stimulus_period + stimulus_period + delay_period


def single_run():
  net = DecisionMaking(scale=1., coherence=-80., mu0=50.)
  runner = bp.DSRunner(
    net, monitors=['A.spike', 'B.spike', 'IA.freq', 'IB.freq']
  )
  runner.run(total_period)

  fig, gs = bp.visualize.get_figure(4, 1, 3, 10)
  axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
  visualize_results(axes, mon=runner.mon)
  plt.show()


def batching_run():
  num_row, num_col = 3, 4
  num_batch = 12
  coherence = bm.expand_dims(bm.linspace(-100, 100., num_batch), 1)

  with bm.batching_environment():
    net = DecisionMaking(scale=1., coherence=coherence, mu0=20.)
    net.reset_state(batch_size=num_batch)
    runner = bp.DSRunner(
      net, monitors=['A.spike', 'B.spike', 'IA.freq', 'IB.freq'], data_first_axis='B'
    )
    runner.run(total_period)

  coherence = bm.as_numpy(coherence)
  fig, gs = bp.visualize.get_figure(num_row, num_col, 3, 4)
  for i in range(num_row):
    for j in range(num_col):
      idx = i * num_col + j
      if idx < num_batch:
        mon = {'A.spike': runner.mon['A.spike'][idx],
               'B.spike': runner.mon['B.spike'][idx],
               'IA.freq': runner.mon['IA.freq'][idx],
               'IB.freq': runner.mon['IB.freq'][idx],
               'ts': runner.mon['ts']}
        ax = fig.add_subplot(gs[i, j])
        visualize_raster(ax, mon=mon, title=f'coherence={coherence[idx, 0]}%')
  plt.show()


if __name__ == '__main__':
  single_run()
  batching_run()
