# %%
import time

# %%
import brainpy.math as bm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import scipy.stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat


# %%
__all__ = [
  'fig1',
  'fig2',
  'fig3',
]


# %%
dt = 1.
time2len = lambda t: int(t / dt)


# %%
def fig1(net, verbose=True):
  # Parameters
  # ----------
  num_rec_train = 30  # Number of learning trials for the recurrent weights
  num_readout_train = 10  # Number of learning trials for the readout weights
  num_perturbation = 5  # Number of perturbation trials

  stimulus_amplitude = 5.0  # Amplitude of the input pulse
  t_offset = time2len(200)  # Time to wait before the stimulation
  d_stim = time2len(50)  # Duration of the stimulation
  d_trajectory = time2len(2000 + 150)  # Duration of the desired target_traj
  t_relax = time2len(550)  # Duration to relax after the target_traj
  trial_duration = t_offset + d_stim + d_trajectory + t_relax  # Total duration of a trial
  times = bm.arange(0, trial_duration, dt)

  perturbation_amplitude = 0.5  # Amplitude of the perturbation pulse
  t_perturb = time2len(500)  # Offset for the perturbation
  d_perturb = time2len(10)  # Duration of the perturbation

  target_baseline = 0.2  # Baseline of the output_traj function
  target_amplitude = 1.  # Maximal value of the output_traj function
  target_width = 30.  # Width of the Gaussian
  target_time = time2len(d_trajectory - 150)  # Peak time within the learning interval

  # Input definitions
  # -----------------

  # Impulse after 200 ms
  impulse = bm.zeros((trial_duration, net.num_input))
  impulse[t_offset:t_offset + d_stim, 0] = stimulus_amplitude

  # Perturbation during the trial
  perturbation = bm.zeros((trial_duration, net.num_input))
  perturbation[t_offset: t_offset + d_stim, 0] = stimulus_amplitude
  perturbation[t_offset + t_perturb: t_offset + t_perturb + d_perturb, 1] = perturbation_amplitude

  # Target output for learning the readout weights
  output_traj = bm.zeros((trial_duration, net.num_output))
  output_traj[:, 0] = target_baseline + (target_amplitude - target_baseline) * \
                      bm.exp(-(t_offset + d_stim + target_time - times) ** 2 / target_width ** 2)

  # Main procedure
  # --------------
  tstart = time.time()

  # Initial trial to determine the innate target_traj
  if verbose: print('Initial trial to determine a target_traj (without noise)')
  initial_traj, initial_output, _ = net.simulate(stimulus=impulse, noise=False)

  # Pre-training test trial
  if verbose: print('Pre-training test trial')
  pretrain_traj, pretrain_output, _ = net.simulate(stimulus=impulse, noise=True)

  # Perturbation trial
  if verbose: print(num_perturbation, 'perturbation trials')
  perturbation_initial = []
  for i in range(num_perturbation):
    _, perturbation_output, _ = net.simulate(stimulus=perturbation, noise=True)
    perturbation_initial.append(perturbation_output)

  # 20 trials of learning for the recurrent weights
  for i in range(num_rec_train):
    t0 = time.time()
    if verbose: print(f'Learning trial recurrent {i + 1} loss: ', end='')
    _, _, loss = net.simulate(stimulus=impulse,
                              target_traj=initial_traj,
                              learn_start=t_offset + d_stim,
                              learn_stop=t_offset + d_stim + d_trajectory)
    if verbose: print(f'{(2 * loss[0] / d_trajectory):5f}, time: {time.time() - t0} s')

  # 10 trials of learning for the readout weights
  for i in range(num_readout_train):
    t0 = time.time()
    if verbose: print(f'Learning trial readout {i + 1} loss: ', end='')
    _, _, loss = net.simulate(stimulus=impulse,
                              target_traj=output_traj,
                              learn_start=t_offset + d_stim,
                              learn_stop=t_offset + d_stim + d_trajectory,
                              learn_readout=True)
    if verbose: print(f'{(2 * loss[0] / d_trajectory):5f}, time: {time.time() - t0} s')

  # Test trial
  if verbose: print('2 test trials')
  reproductions = []
  final_outputs = []
  for i in range(2):
    reproduction, final_output, _ = net.simulate(stimulus=impulse, noise=True)
    reproductions.append(reproduction)
    final_outputs.append(final_output)

  # Perturbation trial
  if verbose: print(num_perturbation, 'perturbation trials')
  perturbation_final = []
  for i in range(num_perturbation):
    _, perturbation_output, _ = net.simulate(stimulus=perturbation)
    perturbation_final.append(perturbation_output)

  if verbose: print('Simulation done in', time.time() - tstart, 'seconds.')

  # Visualization
  # -------------

  plt.figure(figsize=(8, 12))

  # innate trajectory
  ax = plt.subplot2grid((4, 2), (0, 0), colspan=2)
  im = ax.imshow(initial_traj[:, :100].T, aspect='auto', origin='lower')
  cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
  plt.colorbar(im, cax=cax)
  ymin, ymax = ax.get_ylim()
  ax.add_patch(patches.Rectangle((t_offset, ymin), d_stim, ymax - ymin, color='gray', alpha=0.2))
  ax.set_title('Innate Trajectory')
  ax.set_xlabel('Time (ms)')
  ax.set_ylabel('Recurrent units')

  # pre-training results
  ax = plt.subplot2grid((4, 2), (1, 0))
  for i in range(3):
    ax.plot(times, initial_traj[:, i] + i * 2 + 1, 'b')
    ax.plot(times, pretrain_traj[:, i] + i * 2 + 1, 'r')
  ax.set_yticks([i * 2 + 1 for i in range(3)])
  ax.set_yticklabels([0, 0, 0])
  ymin, ymax = ax.get_ylim()
  ax.add_patch(patches.Rectangle((t_offset, ymin), d_stim, ymax - ymin, color='gray', alpha=0.7))
  ax.add_patch(patches.Rectangle((t_offset + d_stim, ymin), d_trajectory, ymax - ymin, color='gray', alpha=0.1))
  ax.set_title('Pre-training')
  ax.set_ylabel('Firing rate $r$ [Hz]')
  ax.set_xlim(times[0], times[-1])

  ax = plt.subplot2grid((4, 2), (2, 0))
  ax.plot(times, initial_output[:, 0], 'b')
  ax.plot(times, pretrain_output[:, 0], 'r')
  ax.axhline(output_traj[0, 0], c='k')
  ax.set_yticks([-2, -1, 0, 1, 2])
  ax.set_ylim((-2, 2))
  ymin, ymax = ax.get_ylim()
  ax.add_patch(patches.Rectangle((t_offset, ymin), d_stim, ymax - ymin, color='gray', alpha=0.7))
  ax.add_patch(patches.Rectangle((t_offset + d_stim, ymin), d_trajectory, ymax - ymin, color='gray', alpha=0.1))
  ax.set_ylabel('Output (test)')
  ax.set_xlim(times[0], times[-1])

  ax = plt.subplot2grid((4, 2), (3, 0))
  for i in range(num_perturbation):
    ax.plot(times, perturbation_initial[i][:, 0])
  ax.axhline(output_traj[0, 0], c='k')
  ax.set_yticks([-2, -1, 0, 1, 2])
  ax.set_ylim((-2, 2))
  ymin, ymax = ax.get_ylim()
  ax.add_patch(patches.Rectangle((t_offset, ymin), d_stim, ymax - ymin, color='gray', alpha=0.7))
  ax.add_patch(patches.Rectangle((t_offset + t_perturb, ymin), d_perturb, ymax - ymin, color='gray', alpha=0.7))
  ax.add_patch(patches.Rectangle((t_offset + d_stim, ymin), d_trajectory, ymax - ymin, color='gray', alpha=0.1))
  ax.set_xlabel('Time (ms)')
  ax.set_ylabel('Output (perturbed)')
  ax.set_xlim(times[0], times[-1])

  # post-training results

  ax = plt.subplot2grid((4, 2), (1, 1))
  for i in range(3):
    ax.plot(times, reproductions[0][:, i] + i * 2 + 1, 'b')
    ax.plot(times, reproductions[1][:, i] + i * 2 + 1, 'r')
  ax.set_yticks([i * 2 + 1 for i in range(3)])
  ax.set_yticklabels([0, 0, 0])
  ymin, ymax = ax.get_ylim()
  ax.add_patch(patches.Rectangle((t_offset, ymin), d_stim, ymax - ymin, color='gray', alpha=0.7))
  ax.add_patch(patches.Rectangle((t_offset + d_stim, ymin), d_trajectory, ymax - ymin, color='gray', alpha=0.1))
  ax.set_title('Post-training')
  ax.set_xlim(times[0], times[-1])

  ax = plt.subplot2grid((4, 2), (2, 1))
  ax.plot(times, final_outputs[0][:, 0], 'b')
  ax.plot(times, final_outputs[1][:, 0], 'r')
  ax.axhline(output_traj[0, 0], c='k')
  ax.set_yticks([-2, -1, 0, 1, 2])
  ax.set_ylim((-2, 2))
  ymin, ymax = ax.get_ylim()
  ax.add_patch(patches.Rectangle((t_offset, ymin), d_stim, ymax - ymin, color='gray', alpha=0.7))
  ax.add_patch(patches.Rectangle((t_offset + d_stim, ymin), d_trajectory, ymax - ymin, color='gray', alpha=0.1))
  ax.set_xlim(times[0], times[-1])

  ax = plt.subplot2grid((4, 2), (3, 1))
  for i in range(num_perturbation):
    ax.plot(times, perturbation_final[i][:, 0])
  ax.axhline(output_traj[0, 0], c='k')
  ax.set_yticks([-2, -1, 0, 1, 2])
  ax.set_ylim((-2, 2))
  ymin, ymax = ax.get_ylim()
  ax.add_patch(patches.Rectangle((t_offset, ymin), d_stim, ymax - ymin, color='gray', alpha=0.7))
  ax.add_patch(patches.Rectangle((t_offset + t_perturb, ymin), d_perturb, ymax - ymin, color='gray', alpha=0.7))
  ax.add_patch(patches.Rectangle((t_offset + d_stim, ymin), d_trajectory, ymax - ymin, color='gray', alpha=0.1))
  ax.set_xlabel('Time (ms)')
  ax.set_xlim(times[0], times[-1])

  plt.tight_layout()
  plt.show()


# %%
def fig2(net, verbose=True):
  # Parameters
  # ----------

  num_rec_train = 30  # Number of learning trials for the recurrent weights
  num_readout_train = 10  # Number of learning trials for the readout weights
  num_test = 5  # Number of test trials
  num_perturb = 5  # Number of perturbation trials

  stimulus_amplitude = 2.0  # Amplitude of the input pulse
  t_offset = time2len(200)  # Time to wait before the stimulation
  d_stim = time2len(50)  # Duration of the stimulation
  t_relax = time2len(150)  # Duration to relax after the target_traj

  perturbation_amplitude = 0.2  # Amplitude of the perturbation pulse
  t_perturb = time2len(300)  # Offset for the perturbation
  d_perturb = time2len(10)  # Duration of the perturbation

  # Input definitions
  # -----------------

  # Retrieve the targets and reformat them
  targets = loadmat('data/DAC_handwriting_output_targets.mat')
  chaos = targets['chaos']
  neuron = targets['neuron']

  # Durations
  _, d_chaos = chaos.shape
  _, d_neuron = neuron.shape

  # Impulses
  impulse_chaos = bm.zeros((t_offset + d_stim + d_chaos + t_relax, net.num_input))
  impulse_chaos[t_offset:t_offset + d_stim, 0] = stimulus_amplitude
  impulse_neuron = bm.zeros((t_offset + d_stim + d_neuron + t_relax, net.num_input))
  impulse_neuron[t_offset:t_offset + d_stim, 2] = stimulus_amplitude

  # Perturbation
  perturbation_chaos = bm.zeros((t_offset + d_stim + d_chaos + t_relax, net.num_input))
  perturbation_chaos[t_offset:t_offset + d_stim, 0] = stimulus_amplitude
  perturbation_chaos[t_offset + t_perturb: t_offset + t_perturb + d_perturb, 1] = perturbation_amplitude
  perturbation_neuron = bm.zeros((t_offset + d_stim + d_neuron + t_relax, net.num_input))
  perturbation_neuron[t_offset:t_offset + d_stim, 2] = stimulus_amplitude
  perturbation_neuron[t_offset + t_perturb: t_offset + t_perturb + d_perturb, 3] = perturbation_amplitude

  # Targets
  target_chaos = bm.zeros((t_offset + d_stim + d_chaos + t_relax, net.num_output))
  target_chaos[t_offset + d_stim: t_offset + d_stim + d_chaos, :] = chaos.T
  target_neuron = bm.zeros((t_offset + d_stim + d_neuron + t_relax, net.num_output))
  target_neuron[t_offset + d_stim: t_offset + d_stim + d_neuron, :] = neuron.T

  # Main procedure
  # --------------
  tstart = time.time()

  # Initial trial to determine the innate target_traj
  if verbose: print('Initial chaos trial')
  trajectory_chaos, initial_chaos_output, _ = net.simulate(stimulus=impulse_chaos, noise=False)
  if verbose: print('Initial neuron trial')
  trajectory_neuron, initial_neuron_output, _ = net.simulate(stimulus=impulse_neuron, noise=False)

  # learning for the recurrent weights
  for i in range(num_rec_train):
    t0 = time.time()
    if verbose: print(f'Learning recurrent {i + 1} "chaos" loss: ', end='')
    _, _, loss = net.simulate(stimulus=impulse_chaos,
                              target_traj=trajectory_chaos,
                              learn_start=t_offset + d_stim,
                              learn_stop=t_offset + d_stim + d_chaos)
    if verbose: print(f'{(2 * loss[0] / d_chaos):.5f} used {(time.time() - t0):5f} s, ', end='')

    t0 = time.time()
    _, _, loss = net.simulate(stimulus=impulse_neuron,
                              target_traj=trajectory_neuron,
                              learn_start=t_offset + d_stim,
                              learn_stop=t_offset + d_stim + d_neuron)
    if verbose: print(f'"neuron" loss: {(2 * loss[0] / d_chaos):5f} used {(time.time() - t0):5f} s')

  # learning for the readout weights
  for i in range(num_readout_train):
    t0 = time.time()
    if verbose: print(f'Learning readout {i + 1} "chaos" loss: ', end='')
    _, _, loss = net.simulate(stimulus=impulse_chaos,
                              target_traj=target_chaos,
                              learn_start=t_offset + d_stim,
                              learn_stop=t_offset + d_stim + d_chaos,
                              learn_readout=True)
    if verbose: print(f'{(2 * loss[0] / d_chaos):.5f} used {(time.time() - t0):5f} s, ', end='')

    t0 = time.time()
    _, _, loss = net.simulate(stimulus=impulse_neuron,
                              target_traj=target_neuron,
                              learn_start=t_offset + d_stim,
                              learn_stop=t_offset + d_stim + d_neuron,
                              learn_readout=True)
    if verbose: print(f'"neuron" loss: {(2 * loss[0] / d_chaos):5f} used {(time.time() - t0):5f} s')

  # Test trials
  final_output_chaos = []
  final_output_neuron = []
  for _ in range(num_test):
    if verbose: print('Test chaos trial')
    _, o, _ = net.simulate(stimulus=impulse_chaos)
    final_output_chaos.append(o)
    if verbose: print('Test neuron trial')
    _, o, _ = net.simulate(stimulus=impulse_neuron)
    final_output_neuron.append(o)

  # Perturbation trials
  perturbation_output_chaos = []
  perturbation_output_neuron = []
  for _ in range(num_perturb):
    if verbose: print('Perturbation chaos trial')
    _, o, _ = net.simulate(stimulus=perturbation_chaos)
    perturbation_output_chaos.append(o)
    if verbose: print('Perturbation neuron trial')
    _, o, _ = net.simulate(stimulus=perturbation_neuron)
    perturbation_output_neuron.append(o)

  if verbose: print('Simulation done in', time.time() - tstart, 'seconds.')

  # Visualization
  # -------------
  subsampling_chaos = (t_offset + d_stim + bm.linspace(0, d_chaos, 20)).astype(bm.int32)
  subsampling_chaos = bm.unique(subsampling_chaos)
  subsampling_neuron = (t_offset + d_stim + bm.linspace(0, d_neuron, 20)).astype(bm.int32)
  subsampling_neuron = bm.unique(subsampling_neuron)

  plt.figure(figsize=(12, 8))
  ax = plt.subplot2grid((2, 2), (0, 0))
  ax.plot(chaos[0, :], chaos[1, :], linewidth=2.)
  for i in range(num_perturb):
    ax.plot(final_output_chaos[i][t_offset + d_stim: t_offset + d_stim + d_chaos, 0],
            final_output_chaos[i][t_offset + d_stim: t_offset + d_stim + d_chaos, 1])
    ax.plot(final_output_chaos[i][subsampling_chaos, 0],
            final_output_chaos[i][subsampling_chaos, 1], 'bo')
  ax.set_xlabel('x')
  ax.set_ylabel('Without perturbation\ny')

  ax = plt.subplot2grid((2, 2), (0, 1))
  ax.plot(neuron[0, :], neuron[1, :], linewidth=2.)
  for i in range(num_perturb):
    ax.plot(final_output_neuron[i][t_offset + d_stim: t_offset + d_stim + d_neuron, 0],
            final_output_neuron[i][t_offset + d_stim: t_offset + d_stim + d_neuron, 1])
    ax.plot(final_output_neuron[i][subsampling_neuron, 0],
            final_output_neuron[i][subsampling_neuron, 1], 'bo')
  ax.set_xlabel('x')
  ax.set_ylabel('y')

  ax = plt.subplot2grid((2, 2), (1, 0))
  ax.plot(chaos[0, :], chaos[1, :], linewidth=2.)
  for i in range(num_perturb):
    ax.plot(perturbation_output_chaos[i][t_offset + d_stim: t_offset + d_stim + d_chaos, 0],
            perturbation_output_chaos[i][t_offset + d_stim: t_offset + d_stim + d_chaos, 1])
    ax.plot(perturbation_output_chaos[i][subsampling_chaos, 0],
            perturbation_output_chaos[i][subsampling_chaos, 1], 'bo')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_ylabel('With perturbation\ny')

  ax = plt.subplot2grid((2, 2), (1, 1))
  ax.plot(neuron[0, :], neuron[1, :], linewidth=2.)
  for i in range(num_perturb):
    ax.plot(perturbation_output_neuron[i][t_offset + d_stim: t_offset + d_stim + d_neuron, 0],
            perturbation_output_neuron[i][t_offset + d_stim: t_offset + d_stim + d_neuron, 1])
    ax.plot(perturbation_output_neuron[i][subsampling_neuron, 0],
            perturbation_output_neuron[i][subsampling_neuron, 1], 'bo')
  ax.set_xlabel('x')
  ax.set_ylabel('y')

  plt.show()


# %%
def fig3(nets, verbose=True):
  # Parameters
  # ----------

  num_rec_train = 20  # Number of learning trials for the recurrent weights
  num_readout_train = 10  # Number of learning trials for the readout weights

  stimulus_amplitude = 5.0  # Amplitude of the input pulse
  t_offset = time2len(200)  # Time to wait before the stimulation
  d_stim = time2len(50)  # Duration of the stimulation
  t_relax = time2len(150)  # Duration to relax after the target_traj

  target_baseline = 0.2  # Baseline of the output_traj function
  target_amplitude = 1.  # Maximal value of the output_traj function
  target_width = 30.  # Width of the Gaussian

  # Main procedure
  # --------------

  # Vary the timing interval
  delays = [250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

  # Store the Pearson correlation coefficients
  pearsons = []

  # Iterate over the delays
  for target_time in delays:
    if verbose:
      print('*' * 60)
      print('Learning a delay of', target_time)
      print('*' * 60)
      print()
    d_trajectory = time2len(target_time + 150)  # Duration of the desired target_traj
    trial_duration = t_offset + d_stim + d_trajectory + t_relax  # Total duration of a trial
    pearsons.append([])

    for n, net in enumerate(nets):  # 10 networks per delay
      if verbose: print(f'##  Network {n + 1}  ##', )
      net.init()

      # Impulse input after 200 ms
      impulse = bm.zeros((trial_duration, net.num_input))
      impulse[t_offset:t_offset + d_stim, 0] = stimulus_amplitude

      # Target output for learning the readout weights
      target = bm.zeros((trial_duration, net.num_output))
      time_axis = bm.linspace(0, trial_duration, trial_duration)
      target[:, 0] = target_baseline + (target_amplitude - target_baseline) * \
                     bm.exp(-(t_offset + d_stim + target_time - time_axis) ** 2 / target_width ** 2)

      # Initial trial to determine the innate target_traj
      if verbose: print('Initial trial to determine a target_traj (without noise)')
      trajectory, initial_output, _ = net.simulate(stimulus=impulse, noise=False)

      # 20 trials of learning for the recurrent weights
      for i in range(num_rec_train):
        t0 = time.time()
        if verbose: print(f'Learning trial recurrent {i + 1} loss: ', end='')
        _, _, loss = net.simulate(stimulus=impulse,
                                  target_traj=trajectory,
                                  learn_start=t_offset + d_stim,
                                  learn_stop=t_offset + d_stim + d_trajectory)
        if verbose: print(f'{(2 * loss[0] / d_trajectory):5f}, time: {time.time() - t0} s')

      # 10 trials of learning for the readout weights
      for i in range(num_readout_train):
        t0 = time.time()
        if verbose: print(f'Learning trial readout {i + 1} loss: ', end='')
        _, _, loss = net.simulate(stimulus=impulse,
                                  target_traj=target,
                                  learn_start=t_offset + d_stim,
                                  learn_stop=t_offset + d_stim + d_trajectory,
                                  learn_readout=True)
        if verbose: print(f'{(2 * loss[0] / d_trajectory):5f}, time: {time.time() - t0} s')

      # Test trial
      if verbose: print('Test trial')
      reproduction, final_output, _ = net.simulate(stimulus=impulse)

      # Pearson correlation coefficient
      pred = final_output[t_offset + d_stim:t_offset + d_stim + d_trajectory, 0]
      desired = target[t_offset + d_stim:t_offset + d_stim + d_trajectory, 0]
      r, p = scipy.stats.pearsonr(desired, pred)
      pearsons[-1].append(r)

  # Save the results
  pearsons = bm.asarray(pearsons)

  # Visualization
  # -------------

  plt.figure(figsize=(8, 6))
  correlation_mean = bm.mean(pearsons ** 2, axis=1)
  correlation_std = bm.std(pearsons ** 2, axis=1)
  plt.errorbar(bm.array(delays) / 1000., correlation_mean, correlation_std / bm.sqrt(10), linestyle='-', marker='^')
  plt.xlim((0., 8.5))
  plt.ylim((-0.1, 1.1))
  plt.xlabel('Interval (s)')
  plt.ylabel('Performance ($R^2$)')
  plt.show()
