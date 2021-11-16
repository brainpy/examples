# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np


__all__ = [
  'Task'
]


class Task(object):
  def __init__(self,
               trial_type='DMS',
               dt=100,  # ms
               tau=100,  # ms
               batch_size=1024):
    self.dt = dt
    self.dt_sec = dt / 1000
    self.tau = tau
    self.alpha = dt / tau
    self.trial_type = trial_type  # DMS, DMRS45, DMRS90, DMRS180, DMC, DMS+DMRS, ABBA, ABCA, dualDMS
    self.batch_size = batch_size

    # input noise
    self.input_mean = 0.0
    self.noise_in = np.sqrt(2 * self.alpha) * 0.1

    # times
    self.dead_time = 0
    self.fix_time = 500
    self.sample_time = 500
    self.delay_time = 1000
    self.test_time = 500
    self.variable_delay_max = 300
    self.mask_duration = 50  # duration of training mask after test onset
    self.rule_onset_time = [self.delay_time]
    self.rule_offset_time = [self.delay_time]

    # rules
    self.catch_trial_pct = 0.0
    self.num_rules = 1  # this will be two for the DMS+DMRS task
    self.test_cost_multiplier = 1.
    self.rule_cue_multiplier = 1.
    self.num_receptive_fields = 1

    # input shape
    self.num_motion_tuned = 24
    self.num_fix_tuned = 0
    self.num_rule_tuned = 0
    self.num_input = self.num_motion_tuned + self.num_fix_tuned + self.num_rule_tuned

    # output shape
    self.num_output = 3

    # others
    self.ABBA_delay = 0

    # other settings
    if trial_type in ['DMS', 'DMC']:
      self.rotation_match = [0]  # angular difference between matching sample and test

    elif trial_type == 'DMRS45':
      self.rotation_match = [45]

    elif trial_type == 'DMRS90':
      self.rotation_match = [90]

    elif trial_type == 'DMRS90ccw':
      self.rotation_match = [-90]

    elif trial_type == 'DMRS180':
      self.rotation_match = [180]

    elif trial_type == 'dualDMS':
      self.rotation_match = [0]
      self.num_motion_tuned = 24 * 2
      self.num_rule_tuned = 6
      self.num_rules = 2
      self.num_receptive_fields = 2
      self.probe_trial_pct = 0
      self.probe_time = 10
      self.analyze_rule = True
      rule_onset_time = [self.dead_time + self.fix_time + self.sample_time + self.delay_time / 2,
                         self.dead_time + self.fix_time + self.sample_time +
                         3 * self.delay_time / 2 + self.test_time]
      self.rule_onset_time = rule_onset_time
      rule_offset_time = [self.dead_time + self.fix_time + self.sample_time +
                          self.delay_time + self.test_time,
                          self.dead_time + self.fix_time + self.sample_time +
                          2 * self.delay_time + 2 * self.test_time]
      self.rule_offset_time = rule_offset_time

    elif trial_type in ['ABBA', 'ABCA']:
      self.rotation_match = [0]
      self.match_test_prob = 0.5
      self.max_num_tests = 3
      self.sample_time = 400
      self.ABBA_delay = 400
      self.delay_time = 6 * self.ABBA_delay
      self.repeat_pct = 0
      self.analyze_test = False
      if trial_type == 'ABBA':
        self.repeat_pct = 0.5

    elif 'DMS+DMRS' in trial_type:
      self.num_rules = 2
      self.num_rule_tuned = 6

      if trial_type == 'DMS+DMRS':
        self.rotation_match = [0, 90]
        self.rule_onset_time = [self.dead_time + self.fix_time + self.sample_time + 500]
        self.rule_offset_time = [self.dead_time + self.fix_time + self.sample_time + 750]

      elif trial_type == 'DMS+DMRS_full_cue':
        self.rotation_match = [0, 90]
        self.rule_onset_time = [self.dead_time]
        self.rule_offset_time = [self.dead_time + self.fix_time + self.sample_time +
                                 self.delay_time + self.test_time]

      else:
        self.rotation_match = [0, 90]
        self.rule_onset_time = [self.dead_time]
        self.rule_offset_time = [self.dead_time + self.fix_time]

    elif trial_type == 'DMS+DMC':
      self.num_rule_tuned = 12
      self.num_rules = 2
      self.rotation_match = [0, 0]
      self.rule_onset_time = [self.dead_time]
      self.rule_offset_time = [self.dead_time + self.fix_time + self.sample_time +
                               self.delay_time + self.test_time]

    elif trial_type == 'DMS+DMRS+DMC':
      self.num_rules = 3
      self.num_rule_tuned = 18
      self.rotation_match = [0, 90, 0]
      self.rule_onset_time = [self.dead_time]
      self.rule_offset_time = [self.dead_time + self.fix_time + self.sample_time +
                               self.delay_time + self.test_time]

    elif trial_type == 'location_DMS':
      self.rotation_match = [0]
      self.num_receptive_fields = 3
      self.num_motion_tuned = 24 * 3
    else:
      raise ValueError(f'{trial_type} is not a recognized trial type.')

    # Length of each trial in time steps
    if trial_type == 'dualDMS':
      self.trial_length = self.dead_time + self.fix_time + self.sample_time + \
                          2 * self.delay_time + 2 * self.test_time
    else:
      self.trial_length = self.dead_time + self.fix_time + self.sample_time + \
                          self.delay_time + self.test_time
    self.num_steps = self.trial_length // dt

    self.dead_time_rng = range(self.delay_time // dt)
    self.sample_time_rng = range((self.dead_time + self.fix_time) // dt,
                                 (self.dead_time + self.fix_time + self.sample_time) // dt)
    self.rule_time_rng = [range(int(self.rule_onset_time[n] / dt), int(self.rule_offset_time[n] / dt))
                          for n in range(len(self.rule_onset_time))]

    # Tuning function data
    self.num_motion_dirs = 8
    self.tuning_height = 4  # magnitude scaling factor for von Mises
    self.kappa = 2  # concentration scaling factor for von Mises

    self.motion_tuning, self.fix_tuning, self.rule_tuning = self._create_tuning_functions()

  def generate_trial(self, test_mode=False, set_rule=None):
    if self.trial_type in ['DMS', 'DMRS45', 'DMRS90', 'DMRS90ccw', 'DMRS180', 'DMC',
                           'DMS+DMRS', 'DMS+DMRS_early_cue', 'DMS+DMRS_full_cue',
                           'DMS+DMC', 'DMS+DMRS+DMC', 'location_DMS']:
      trial_info = self._generate_basic_trial(test_mode, set_rule)
    elif self.trial_type in ['ABBA', 'ABCA']:
      trial_info = self._generate_ABBA_trial(test_mode)
    elif self.trial_type == 'dualDMS':
      trial_info = self._generate_dualDMS_trial(test_mode)
    else:
      raise ValueError
    # input activity needs to be non-negative
    trial_info['neural_input'] = np.maximum(0., trial_info['neural_input'])
    return trial_info

  def plot_neural_input(self, trial_info):
    print(trial_info['desired_output'][:, 0, :].T)
    f = plt.figure(figsize=(8, 4))
    ax = f.add_subplot(1, 1, 1)

    t = np.arange(0, 400 + 500 + 2000, self.dt)
    t -= 900
    t0, t1, t2, t3 = np.where(t == -500), np.where(t == 0), np.where(t == 500), np.where(t == 1500)
    # im = ax.imshow(trial_info['neural_input'][:,0,:].T, aspect='auto', interpolation='none')
    im = ax.imshow(trial_info['neural_input'][:, :, 0], aspect='auto', interpolation='none')
    # plt.imshow(trial_info['desired_output'][:, :, 0], aspect='auto')
    # ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
    ax.set_xticklabels([-500, 0, 500, 1500])
    ax.set_yticks([0, 9, 18, 27])
    ax.set_yticklabels([0, 90, 180, 270])
    f.colorbar(im, orientation='vertical')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Motion direction')
    ax.set_xlabel('Time relative to sample onset (ms)')
    ax.set_title('Motion input')
    plt.show()

  def _generate_basic_trial(self, test_mode, set_rule=None):
    """ Generate a delayed matching task.

    Goal is to determine whether the sample stimulus, possibly manipulated by a rule,
    is identical to a test stimulus. Sample and test stimuli are separated by a delay.
    """
    # duration of mask after test onset
    mask_duration = self.mask_duration // self.dt

    data = {
      'desired_output': np.zeros((self.num_steps, self.batch_size, self.num_output), dtype=np.float32),
      'train_mask': np.ones((self.num_steps, self.batch_size), dtype=np.float32),
      'sample': np.zeros((self.batch_size,), dtype=np.int8),
      'test': np.zeros((self.batch_size,), dtype=np.int8),
      'rule': np.zeros((self.batch_size,), dtype=np.int8),
      'match': np.zeros((self.batch_size,), dtype=np.int8),
      'catch': np.zeros((self.batch_size,), dtype=np.int8),
      'probe': np.zeros((self.batch_size,), dtype=np.int8),
      'neural_input': np.random.normal(self.input_mean, self.noise_in, size=(self.num_steps, self.batch_size, self.num_input))
    }

    # set to mask equal to zero during the dead time
    data['train_mask'][:int(self.dead_time // self.dt), :] = 0.

    for t in range(self.batch_size):
      sample_dir = np.random.randint(self.num_motion_dirs)
      test_RF = np.random.choice([1, 2]) if self.trial_type == 'location_DMS' else 0
      rule = np.random.randint(self.num_rules) if set_rule is None else set_rule
      if self.trial_type == 'DMC' or \
          (self.trial_type == 'DMS+DMC' and rule == 1) or \
          (self.trial_type == 'DMS+DMRS+DMC' and rule == 2):
        # for DMS+DMC trial type, rule 0 will be DMS, and rule 1 will be DMC
        current_trial_DMC = True
      else:
        current_trial_DMC = False
      match = np.random.randint(2)
      catch = np.random.rand() < self.catch_trial_pct

      # rotation
      match_rotation = int(self.num_motion_dirs * self.rotation_match[rule] / 360)

      # Determine the delay time for this trial. The total trial length
      # is kept constant, so a shorter delay implies a longer test stimulus
      test_onset = (self.dead_time + self.fix_time + self.sample_time + self.delay_time) // self.dt

      test_time_rng = range(test_onset, self.num_steps)
      fix_time_rng = range(test_onset)
      data['train_mask'][test_onset:test_onset + mask_duration, t] = 0.

      # Generate the sample and test stimuli based on the rule
      if not test_mode:
        # DMC
        if current_trial_DMC:  # categorize between two equal size, contiguous zones
          sample_cat = np.floor(sample_dir / (self.num_motion_dirs / 2))
          if match == 1:  # match trial
            # do not use sample_dir as a match test stimulus
            dir0 = int(sample_cat * self.num_motion_dirs // 2)
            dir1 = int(self.num_motion_dirs // 2 + sample_cat * self.num_motion_dirs // 2)
            possible_dirs = list(range(dir0, dir1))
            test_dir = possible_dirs[np.random.randint(len(possible_dirs))]
          else:
            test_dir = sample_cat * (self.num_motion_dirs // 2) + np.random.randint(self.num_motion_dirs // 2)
            test_dir = np.int_((test_dir + self.num_motion_dirs // 2) % self.num_motion_dirs)
        # DMS or DMRS
        else:
          matching_dir = (sample_dir + match_rotation) % self.num_motion_dirs
          if match == 1:  # match trial
            test_dir = matching_dir
          else:
            possible_dirs = np.setdiff1d(list(range(self.num_motion_dirs)), matching_dir)
            test_dir = possible_dirs[np.random.randint(len(possible_dirs))]
      else:
        test_dir = np.random.randint(self.num_motion_dirs)
        # this next part only working for DMS, DMRS tasks
        matching_dir = (sample_dir + match_rotation) % self.num_motion_dirs
        match = 1 if test_dir == matching_dir else 0

      # Calculate neural input based on sample, tests, fixation, rule, and probe
      # SAMPLE stimulus
      data['neural_input'][self.sample_time_rng, t, :] += \
        np.reshape(self.motion_tuning[:, 0, sample_dir], (1, -1))

      # TEST stimulus
      if not catch:
        data['neural_input'][test_time_rng, t, :] += \
          np.reshape(self.motion_tuning[:, test_RF, test_dir], (1, -1))

      # FIXATION cue
      if self.num_fix_tuned > 0:
        data['neural_input'][fix_time_rng, t] += np.reshape(self.fix_tuning[:, 0], (-1, 1))

      # RULE CUE
      if self.num_rules > 1 and self.num_rule_tuned > 0:
        data['neural_input'][self.rule_time_rng[0], t, :] += \
          np.reshape(self.rule_tuning[:, rule], (1, -1))

      # Determine the desired network output response
      data['desired_output'][fix_time_rng, t, 0] = 1.
      if not catch:
        # can use a greater weight for test period if needed
        data['train_mask'][test_time_rng, t] *= self.test_cost_multiplier
        if match == 0:
          data['desired_output'][test_time_rng, t, 1] = 1.
        else:
          data['desired_output'][test_time_rng, t, 2] = 1.
      else:
        data['desired_output'][test_time_rng, t, 0] = 1.

      # Append trial info
      data['sample'][t] = sample_dir
      data['test'][t] = test_dir
      data['rule'][t] = rule
      data['catch'][t] = catch
      data['match'][t] = match

    return data

  def _generate_ABBA_trial(self, test_mode):
    """Generate ABBA trials.

    Sample stimulis is followed by up to max_num_tests test stimuli.
    The goal is to to indicate when a test stimulus matches the sample,
    """

    # duration of mask after test onset
    mask_duration = self.mask_duration // self.dt

    # only one receptive field in this task
    RF = 0

    ABBA_delay = self.ABBA_delay // self.dt
    eos = (self.dead_time + self.fix_time + self.ABBA_delay) // self.dt
    test_time_rng = []
    mask_time_rng = []
    for n in range(self.max_num_tests):
      test_time_rng.append(range(eos + ABBA_delay * (2 * n + 1), eos + ABBA_delay * (2 * n + 2)))
      mask_time_rng.append(range(eos + ABBA_delay * (2 * n + 1), eos + ABBA_delay * (2 * n + 1) + mask_duration))

    data = {
      'desired_output': np.zeros((self.num_steps, self.batch_size, self.num_output), dtype=np.float32),
      'train_mask': np.ones((self.num_steps, self.batch_size), dtype=np.float32),
      'sample': np.zeros((self.batch_size,), dtype=np.float32),
      'test': -1 * np.ones((self.batch_size, self.max_num_tests), dtype=np.float32),
      'rule': np.zeros((self.batch_size,), dtype=np.int8),
      'match': np.zeros((self.batch_size, self.max_num_tests), dtype=np.int8),
      'catch': np.zeros((self.batch_size,), dtype=np.int8),
      'probe': np.zeros((self.batch_size,), dtype=np.int8),
      'num_test_stim': np.zeros((self.batch_size,), dtype=np.int8),
      'repeat_test_stim': np.zeros((self.batch_size,), dtype=np.int8),
      'neural_input': np.random.normal(self.input_mean, self.noise_in,
                                       size=(self.num_steps, self.batch_size, self.num_input))
    }

    # set to mask equal to zero during the dead time
    data['train_mask'][self.dead_time_rng, :] = 0

    # set fixation equal to 1 for all times; will then change
    data['desired_output'][:, :, 0] = 1

    for t in range(self.batch_size):
      # generate trial params
      sample_dir = np.random.randint(self.num_motion_dirs)

      # Generate up to 'max_num_tests' test stimuli
      # Sequential test stimuli are identical with probability repeat_pct
      stim_dirs = [sample_dir]

      if test_mode:
        # used to analyze how sample and test neuronal and synaptic tuning relate
        # not used to evaluate task accuracy
        while len(stim_dirs) <= self.max_num_tests:
          q = np.random.randint(self.num_motion_dirs)
          stim_dirs.append(q)
      else:
        while len(stim_dirs) <= self.max_num_tests:
          if np.random.rand() < self.match_test_prob:
            stim_dirs.append(sample_dir)
          else:
            if len(stim_dirs) > 1 and np.random.rand() < self.repeat_pct:
              # repeat last stimulus
              stim_dirs.append(stim_dirs[-1])
              data['repeat_test_stim'][t] = 1
            else:
              possible_dirs = np.setdiff1d(list(range(self.num_motion_dirs)), [stim_dirs])
              distractor_dir = possible_dirs[np.random.randint(len(possible_dirs))]
              stim_dirs.append(distractor_dir)

      data['num_test_stim'][t] = len(stim_dirs)

      # Calculate input neural activity based on trial params
      # SAMPLE stimuli
      data['neural_input'][self.sample_time_rng, t, :] += \
        np.reshape(self.motion_tuning[:, RF, sample_dir], (1, -1))

      # TEST stimuli
      # first element of stim_dirs is the original sample stimulus
      for i, stim_dir in enumerate(stim_dirs[1:]):
        data['test'][t, i] = stim_dir
        # test_time_rng = range(eos+(2*i+1)*ABBA_delay, eos+(2*i+2)*ABBA_delay)
        data['neural_input'][test_time_rng[i], t, :] += \
          np.reshape(self.motion_tuning[:, RF, stim_dir], (1, -1))
        data['train_mask'][mask_time_rng[i], t] = 0
        data['desired_output'][test_time_rng[i], t, 0] = 0
        # can use a greater weight for test period if needed
        data['train_mask'][test_time_rng[i], t] *= self.test_cost_multiplier
        if stim_dir == sample_dir:
          data['desired_output'][test_time_rng[i], t, 2] = 1
          data['match'][t, i] = 1
        else:
          data['desired_output'][test_time_rng[i], t, 1] = 1

      data['sample'][t] = sample_dir

    return data

  def _generate_dualDMS_trial(self, test_mode):
    """Generate a trial based on "Reactivation of latent working memories
    with transcranial magnetic stimulation".

    Trial outline
    1. Dead period
    2. Fixation
    3. Two sample stimuli presented
    4. Delay (cue in middle, and possibly probe later)
    5. Test stimulus (to cued modality, match or non-match)
    6. Delay (cue in middle, and possibly probe later)
    7. Test stimulus

    INPUTS:
    1. sample_time (duration of sample stimulus)
    2. test_time
    3. delay_time
    4. cue_time (duration of rule cue, always presented halfway during delay)
    5. probe_time (usually set to one time step, always presented 3/4 through delay
    """

    test_time_rng = []
    mask_time_rng = []
    fix_time_rng = []

    for n in range(2):
      t1 = self.dead_time + self.fix_time + self.sample_time + (n + 1) * self.delay_time + n * self.test_time
      test_time_rng.append(range(t1 // self.dt, (t1 + self.test_time) // self.dt))
      mask_time_rng.append(range(t1 // self.dt, (t1 + self.mask_duration) // self.dt))

    t2 = self.dead_time + self.fix_time + self.sample_time + self.delay_time + self.test_time
    fix_time_rng.append(range(self.dead_time // self.dt, t2 // self.dt))
    fix_time_rng.append(range(t2 // self.dt, (t2 + self.delay_time) // self.dt))

    data = {
      'desired_output': np.zeros((self.num_steps, self.batch_size, self.num_output), dtype=np.float32),
      'train_mask': np.ones((self.num_steps, self.batch_size), dtype=np.float32),
      'sample': np.zeros((self.batch_size, 2), dtype=np.int8),
      'test': np.zeros((self.batch_size, 2, 2), dtype=np.int8),
      'test_mod': np.zeros((self.batch_size, 2), dtype=np.int8),
      'rule': np.zeros((self.batch_size, 2), dtype=np.int8),
      'match': np.zeros((self.batch_size, 2), dtype=np.int8),
      'catch': np.zeros((self.batch_size, 2), dtype=np.int8),
      'probe': np.zeros((self.batch_size, 2), dtype=np.int8),
      'neural_input': np.random.normal(self.input_mean, self.noise_in,
                                       size=(self.num_steps, self.batch_size, self.num_input))
    }

    for t in range(self.batch_size):

      # generate sample, match, rule and prob params
      for i in range(2):
        data['sample'][t, i] = np.random.randint(self.num_motion_dirs)
        data['match'][t, i] = np.random.randint(2)
        data['rule'][t, i] = np.random.randint(2)
        data['catch'][t, i] = np.random.rand() < self.catch_trial_pct
        if i == 1:
          # only generate a pulse during 2nd delay epoch
          data['probe'][t, i] = np.random.rand() < self.probe_trial_pct

      # determine test stimulus based on sample and match status
      for i in range(2):
        if test_mode:
          data['test'][t, i, 0] = np.random.randint(self.num_motion_dirs)
          data['test'][t, i, 1] = np.random.randint(self.num_motion_dirs)
        else:
          # if trial is not a catch, the upcoming test modality (what the network
          # should be attending to) is given by the rule cue
          if not data['catch'][t, i]:
            data['test_mod'][t, i] = data['rule'][t, i]
          else:
            data['test_mod'][t, i] = (data['rule'][t, i] + 1) % 2

          # cued test stimulus
          if data['match'][t, i] == 1:
            data['test'][t, i, 0] = data['sample'][t, data['test_mod'][t, i]]
          else:
            sample = data['sample'][t, data['test_mod'][t, i]]
            bad_directions = [sample]
            possible_stim = np.setdiff1d(list(range(self.num_motion_dirs)), bad_directions)
            data['test'][t, i, 0] = possible_stim[np.random.randint(len(possible_stim))]

          # non-cued test stimulus
          data['test'][t, i, 1] = np.random.randint(self.num_motion_dirs)

      # Calculate input neural activity based on trial params #
      # SAMPLE stimuli
      data['neural_input'][self.sample_time_rng, t, :] += \
        np.reshape(self.motion_tuning[:, 0, data['sample'][t, 0]], (1, -1))
      data['neural_input'][self.sample_time_rng, t, :] += \
        np.reshape(self.motion_tuning[:, 1, data['sample'][t, 1]], (1, -1))

      # Cued TEST stimuli
      data['neural_input'][test_time_rng[0], t, :] += np.reshape(
        self.motion_tuning[:, data['test_mod'][t, 0], data['test'][t, 0, 0]], (1, -1))
      data['neural_input'][test_time_rng[1], t, :] += np.reshape(
        self.motion_tuning[:, data['test_mod'][t, 1], data['test'][t, 1, 0]], (1, -1))

      # Non-cued TEST stimuli
      data['neural_input'][test_time_rng[0], t, :] += np.reshape(
        self.motion_tuning[:, (1 + data['test_mod'][t, 0]) % 2, data['test'][t, 0, 1]], (1, -1))
      data['neural_input'][test_time_rng[1], t, :] += np.reshape(
        self.motion_tuning[:, (1 + data['test_mod'][t, 1]) % 2, data['test'][t, 1, 1]], (1, -1))

      # FIXATION
      data['neural_input'][fix_time_rng[0], t, :] += np.reshape(self.fix_tuning[:, 0], (1, -1))
      data['neural_input'][fix_time_rng[1], t, :] += np.reshape(self.fix_tuning[:, 0], (1, -1))

      # RULE CUE
      data['neural_input'][self.rule_time_rng[0], t, :] += \
        np.reshape(self.rule_tuning[:, data['rule'][t, 0]], (1, -1))
      data['neural_input'][self.rule_time_rng[1], t, :] += \
        np.reshape(self.rule_tuning[:, data['rule'][t, 1]], (1, -1))

      # Desired outputs #
      # FIXATION
      data['desired_output'][fix_time_rng[0], t, 0] = 1
      data['desired_output'][fix_time_rng[1], t, 0] = 1
      # TEST 1
      # can use a greater weight for test period if needed
      data['train_mask'][test_time_rng[0], t] *= self.test_cost_multiplier
      if data['match'][t, 0] == 1:
        data['desired_output'][test_time_rng[0], t, 2] = 1
      else:
        data['desired_output'][test_time_rng[0], t, 1] = 1
      # TEST 2
      # can use a greater weight for test period if needed
      data['train_mask'][test_time_rng[1], t] *= self.test_cost_multiplier
      if data['match'][t, 1] == 1:
        data['desired_output'][test_time_rng[1], t, 2] = 1
      else:
        data['desired_output'][test_time_rng[1], t, 1] = 1

      # set to mask equal to zero during the dead time, and during the first times of test stimuli
      data['train_mask'][:self.dead_time // self.dt, t] = 0
      data['train_mask'][mask_time_rng[0], t] = 0
      data['train_mask'][mask_time_rng[1], t] = 0

    return data

  def _create_tuning_functions(self):
    motion_tuning = np.zeros((self.num_input, self.num_receptive_fields, self.num_motion_dirs))
    fix_tuning = np.zeros((self.num_input, 1))
    rule_tuning = np.zeros((self.num_input, self.num_rules))

    # Generate list of preferred directions
    # dividing neurons by 2 since two equal
    # groups representing two modalities
    pref_dirs = np.arange(0, 360, 360 / (self.num_motion_tuned // self.num_receptive_fields))

    # Generate list of possible stimulus directions
    stim_dirs = np.arange(0, 360, 360 / self.num_motion_dirs)

    for n in range(self.num_motion_tuned // self.num_receptive_fields):
      for i in range(self.num_motion_dirs):
        for r in range(self.num_receptive_fields):
          if self.trial_type == 'distractor':
            if n % self.num_motion_dirs == i:
              motion_tuning[n, 0, i] = self.tuning_height
          else:
            d = np.cos((stim_dirs[i] - pref_dirs[n]) / 180 * np.pi)
            n_ind = n + r * self.num_motion_tuned // self.num_receptive_fields
            motion_tuning[n_ind, r, i] = self.tuning_height * np.exp(self.kappa * d) / np.exp(self.kappa)

    for n in range(self.num_fix_tuned):
      fix_tuning[self.num_motion_tuned + n, 0] = self.tuning_height

    for n in range(self.num_rule_tuned):
      for i in range(self.num_rules):
        if n % self.num_rules == i:
          rule_tuning[self.num_motion_tuned + self.num_fix_tuned + n, i] = \
            self.tuning_height * self.rule_cue_multiplier

    return motion_tuning, fix_tuning, rule_tuning
