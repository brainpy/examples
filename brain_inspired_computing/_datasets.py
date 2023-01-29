import datetime as dt
import os

import brainpy_datasets as bd
import numpy as np
import pandas as pd


__all__= [
  'HarData',
  'CheetahData',
  'GestureData',
  'OccupancyData',
  'OzoneData',
  'PersonData',
  'PowerData',
  'SMnistData',
  'TrafficData',
]

def _cut_in_sequences(x, seq_len, inc=1, y=None, stack_axis=None):
  sequences_x = []
  sequences_y = []
  for s in range(0, x.shape[0] - seq_len - 1, inc):
    end = s + seq_len
    sequences_x.append(x[s:end])
    if y is None:
      sequences_y.append(x[s + 1:end + 1])
    else:
      sequences_y.append(y[s:end])
  if stack_axis is None:
    return sequences_x, sequences_y
  else:
    return np.stack(sequences_x, axis=stack_axis), np.stack(sequences_y, axis=stack_axis)


def _to_float(v):
  if (v == "?"):
    return 0
  else:
    return float(v)

def _convert_to_floats(feature_col, memory):
  for i in range(len(feature_col)):
    if (feature_col[i] == "?" or feature_col[i] == "\n"):
      feature_col[i] = memory[i]
    else:
      feature_col[i] = float(feature_col[i])
      memory[i] = feature_col[i]
  return feature_col, memory


def _one_hot(x, n):
  y = np.zeros(n, dtype=np.float32)
  y[x] = 1
  return y


class HarData:
  """
  Human Activity Recognition database

  The experiments have been carried out with a group of 30 volunteers
  within an age bracket of 19-48 years. Each person performed six activities
  (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
  wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded
  accelerometer and gyroscope, we captured 3-axial linear acceleration and
  3-axial angular velocity at a constant rate of 50Hz. The experiments have
  been video-recorded to label the data manually. The obtained dataset has
  been randomly partitioned into two sets, where 70% of the volunteers was
  selected for generating the training data and 30% the test data.

  The sensor signals (accelerometer and gyroscope) were pre-processed by
  applying noise filters and then sampled in fixed-width sliding windows
  of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration
  signal, which has gravitational and body motion components, was separated
  using a Butterworth low-pass filter into body acceleration and gravity.
  The gravitational force is assumed to have only low frequency components,
  therefore a filter with 0.3 Hz cutoff frequency was used. From each window,
  a vector of features was obtained by calculating variables from the time
  and frequency domain.

  https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
  """

  def __init__(self, seq_len=16):
    train_x = np.loadtxt("data/har/UCI HAR Dataset/train/X_train.txt")
    train_y = (np.loadtxt("data/har/UCI HAR Dataset/train/y_train.txt") - 1).astype(np.int32)
    test_x = np.loadtxt("data/har/UCI HAR Dataset/test/X_test.txt")
    test_y = (np.loadtxt("data/har/UCI HAR Dataset/test/y_test.txt") - 1).astype(np.int32)

    train_x, train_y = _cut_in_sequences(train_x, seq_len, y=train_y, stack_axis=1)
    test_x, test_y = _cut_in_sequences(test_x, seq_len, inc=8, y=test_y, stack_axis=1)
    print("Total number of training sequences: {}".format(train_x.shape[1]))
    permutation = np.random.RandomState(893429).permutation(train_x.shape[1])
    valid_size = int(0.1 * train_x.shape[1])
    print("Validation split: {}, training split: {}".format(valid_size, train_x.shape[1] - valid_size))

    self.valid_x = train_x[:, permutation[:valid_size]]
    self.valid_y = train_y[:, permutation[:valid_size]]
    self.train_x = train_x[:, permutation[valid_size:]]
    self.train_y = train_y[:, permutation[valid_size:]]

    self.test_x = test_x
    self.test_y = test_y
    print("Total number of test sequences: {}".format(self.test_x.shape[1]))

  def iterate_train(self, batch_size=16):
    total_seqs = self.train_x.shape[1]
    permutation = np.random.permutation(total_seqs)
    total_batches = total_seqs // batch_size

    for i in range(total_batches):
      start = i * batch_size
      end = start + batch_size
      batch_x = self.train_x[:, permutation[start:end]]
      batch_y = self.train_y[:, permutation[start:end]]
      yield (batch_x, batch_y)


class CheetahData:
  def __init__(self, seq_len=32):
    all_files = sorted([os.path.join("data/cheetah", d) for d in os.listdir("data/cheetah")
                        if d.endswith(".npy")])

    train_files = all_files[15:25]
    test_files = all_files[5:15]
    valid_files = all_files[:5]

    self.seq_len = seq_len
    self.obs_size = 17

    self.train_x, self.train_y = self._load_files(train_files)
    self.test_x, self.test_y = self._load_files(test_files)
    self.valid_x, self.valid_y = self._load_files(valid_files)

    print("train_x.shape:", str(self.train_x.shape))
    print("train_y.shape:", str(self.train_y.shape))
    print("valid_x.shape:", str(self.valid_x.shape))
    print("valid_y.shape:", str(self.valid_y.shape))
    print("test_x.shape:", str(self.test_x.shape))
    print("test_y.shape:", str(self.test_y.shape))

  def _load_files(self, files):
    all_x = []
    all_y = []
    for f in files:
      arr = np.load(f)
      arr = arr.astype(np.float32)
      x, y = _cut_in_sequences(arr, self.seq_len, 10)
      all_x.extend(x)
      all_y.extend(y)
    return np.stack(all_x, axis=1), np.stack(all_y, axis=1)

  def iterate_train(self, batch_size=16):
    total_seqs = self.train_x.shape[1]
    permutation = np.random.permutation(total_seqs)
    total_batches = total_seqs // batch_size

    for i in range(total_batches):
      start = i * batch_size
      end = start + batch_size
      batch_x = self.train_x[:, permutation[start:end]]
      batch_y = self.train_y[:, permutation[start:end]]
      yield (batch_x, batch_y)


class GestureData:
  """
  The dataset is composed by features extracted from 7 videos with
  people gesticulating, aiming at studying Gesture Phase Segmentation.

  Each video is represented by two files: a raw file, which contains the
  position of hands, wrists, head and spine of the user in each frame;
  and a processed file, which contains velocity and acceleration of hands
  and wrists. See the data set description for more information on the dataset.

  https://archive.ics.uci.edu/ml/datasets/Gesture+Phase+Segmentation
  """

  training_files = [
    "a3_va3.csv",
    "b1_va3.csv",
    "b3_va3.csv",
    "c1_va3.csv",
    "c3_va3.csv",
    "a2_va3.csv",
    "a1_va3.csv",
  ]

  def __init__(self, seq_len=32):
    train_traces = []
    interleaved_train = True
    for f in self.training_files:
      train_traces.extend(self.cut_in_sequences(self.load_trace(os.path.join("data/gesture", f)),
                                                seq_len, interleaved=interleaved_train))

    train_x, train_y = list(zip(*train_traces))
    self.train_x = np.stack(train_x, axis=1)
    self.train_y = np.stack(train_y, axis=1)

    flat_x = self.train_x.reshape([-1, self.train_x.shape[-1]])
    mean_x = np.mean(flat_x, axis=0)
    std_x = np.std(flat_x, axis=0)
    self.train_x = (self.train_x - mean_x) / std_x

    total_seqs = self.train_x.shape[1]
    print("Total number of training sequences: {}".format(total_seqs))
    permutation = np.random.RandomState(23489).permutation(total_seqs)
    valid_size = int(0.1 * total_seqs)
    test_size = int(0.15 * total_seqs)

    self.valid_x = self.train_x[:, permutation[:valid_size]]
    self.valid_y = self.train_y[:, permutation[:valid_size]]
    self.test_x = self.train_x[:, permutation[valid_size:valid_size + test_size]]
    self.test_y = self.train_y[:, permutation[valid_size:valid_size + test_size]]
    self.train_x = self.train_x[:, permutation[valid_size + test_size:]]
    self.train_y = self.train_y[:, permutation[valid_size + test_size:]]

  def load_trace(self, filename):
    df = pd.read_csv(filename, header=0)
    str_y = df["Phase"].values
    convert = {"D": 0, "P": 1, "S": 2, "H": 3, "R": 4}
    y = np.empty(str_y.shape[0], dtype=np.int32)
    for i in range(str_y.shape[0]):
      y[i] = convert[str_y[i]]
    x = df.values[:, :-1].astype(np.float32)
    return (x, y)

  def cut_in_sequences(self, tup, seq_len, interleaved=False):
    x, y = tup
    num_sequences = x.shape[0] // seq_len
    sequences = []
    for s in range(num_sequences):
      start = seq_len * s
      end = start + seq_len
      sequences.append((x[start:end], y[start:end]))
      if interleaved and s < num_sequences - 1:
        start += seq_len // 2
        end = start + seq_len
        sequences.append((x[start:end], y[start:end]))
    return sequences

  def iterate_train(self, batch_size=16):
    total_seqs = self.train_x.shape[1]
    permutation = np.random.permutation(total_seqs)
    total_batches = total_seqs // batch_size

    for i in range(total_batches):
      start = i * batch_size
      end = start + batch_size
      batch_x = self.train_x[:, permutation[start:end]]
      batch_y = self.train_y[:, permutation[start:end]]
      yield (batch_x, batch_y)


class OccupancyData:
  """

  https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+
  """

  def __init__(self, seq_len=16):
    train_x, train_y = self.read_file("data/occupancy/datatraining.txt")
    test0_x, test0_y = self.read_file("data/occupancy/datatest.txt")
    test1_x, test1_y = self.read_file("data/occupancy/datatest2.txt")

    mean_x = np.mean(train_x, axis=0)
    std_x = np.std(train_x, axis=0)
    train_x = (train_x - mean_x) / std_x
    test0_x = (test0_x - mean_x) / std_x
    test1_x = (test1_x - mean_x) / std_x

    train_x, train_y = _cut_in_sequences(train_x, seq_len, y=train_y, stack_axis=1)
    test0_x, test0_y = _cut_in_sequences(test0_x, seq_len, inc=8, y=test0_y, stack_axis=1)
    test1_x, test1_y = _cut_in_sequences(test1_x, seq_len, inc=8, y=test1_y, stack_axis=1)
    print("Total number of training sequences: {}".format(train_x.shape[1]))
    permutation = np.random.RandomState(893429).permutation(train_x.shape[1])
    valid_size = int(0.1 * train_x.shape[1])
    print("Validation split: {}, training split: {}".format(valid_size, train_x.shape[1] - valid_size))

    self.valid_x = train_x[:, permutation[:valid_size]]
    self.valid_y = train_y[:, permutation[:valid_size]]
    self.train_x = train_x[:, permutation[valid_size:]]
    self.train_y = train_y[:, permutation[valid_size:]]

    self.test_x = np.concatenate([test0_x, test1_x], axis=1)
    self.test_y = np.concatenate([test0_y, test1_y], axis=1)
    print("Total number of test sequences: {}".format(self.test_x.shape[1]))

  def read_file(self, filename):
    df = pd.read_csv(filename)
    data_x = np.stack(
      [df['Temperature'].values,
       df['Humidity'].values,
       df['Light'].values,
       df['CO2'].values,
       df['HumidityRatio'].values, ],
      axis=-1
    )
    data_y = df['Occupancy'].values.astype(np.int32)
    return data_x, data_y

  def iterate_train(self, batch_size=16):
    total_seqs = self.train_x.shape[1]
    permutation = np.random.permutation(total_seqs)
    total_batches = total_seqs // batch_size

    for i in range(total_batches):
      start = i * batch_size
      end = start + batch_size
      batch_x = self.train_x[:, permutation[start:end]]
      batch_y = self.train_y[:, permutation[start:end]]
      yield (batch_x, batch_y)


class OzoneData:
  """

  https://archive.ics.uci.edu/ml/datasets/Ozone+Level+Detection
  """

  def __init__(self, seq_len=32):
    all_x = []
    all_y = []

    with open("data/ozone/eighthr.data", "r") as f:
      miss = 0
      total = 0
      while True:
        line = f.readline()
        if (line is None):
          break
        line = line[:-1]
        parts = line.split(',')

        total += 1
        for i in range(1, len(parts) - 1):
          if (parts[i] == "?"):
            miss += 1
            break

        if (len(parts) != 74):
          break
        label = int(float(parts[-1]))
        feats = [_to_float(parts[i]) for i in range(1, len(parts) - 1)]

        all_x.append(np.array(feats))
        all_y.append(label)
    print("Missing features in {} out of {} samples ({:0.2f})".format(miss, total, 100 * miss / total))
    print("Read {} lines".format(len(all_x)))
    all_x = np.stack(all_x, axis=0)
    all_y = np.array(all_y)
    print("Imbalance: {:0.2f}%".format(100 * np.mean(all_y)))
    all_x -= np.mean(all_x)  # normalize
    all_x /= np.std(all_x)  # normalize

    x, y = all_x, all_y
    train_x, train_y = _cut_in_sequences(x, seq_len, inc=4, y=y)

    self.train_x = np.stack(train_x, axis=1)
    self.train_y = np.stack(train_y, axis=1)

    total_seqs = self.train_x.shape[1]
    print("Total number of training sequences: {}".format(total_seqs))
    permutation = np.random.RandomState(23489).permutation(total_seqs)
    valid_size = int(0.1 * total_seqs)
    test_size = int(0.15 * total_seqs)

    self.valid_x = self.train_x[:, permutation[:valid_size]]
    self.valid_y = self.train_y[:, permutation[:valid_size]]
    self.test_x = self.train_x[:, permutation[valid_size:valid_size + test_size]]
    self.test_y = self.train_y[:, permutation[valid_size:valid_size + test_size]]
    self.train_x = self.train_x[:, permutation[valid_size + test_size:]]
    self.train_y = self.train_y[:, permutation[valid_size + test_size:]]

  def iterate_train(self, batch_size=16):
    total_seqs = self.train_x.shape[1]
    permutation = np.random.permutation(total_seqs)
    total_batches = total_seqs // batch_size

    for i in range(total_batches):
      start = i * batch_size
      end = start + batch_size
      batch_x = self.train_x[:, permutation[start:end]]
      batch_y = self.train_y[:, permutation[start:end]]
      yield (batch_x, batch_y)




class PersonData:
  class_map = {
    'lying down': 0,
    'lying': 0,
    'sitting down': 1,
    'sitting': 1,
    'standing up from lying': 2,
    'standing up from sitting': 2,
    'standing up from sitting on the ground': 2,
    "walking": 3,
    "falling": 4,
    'on all fours': 5,
    'sitting on the ground': 6,
  }  # 11 to 7

  sensor_ids = {
    "010-000-024-033": 0,
    "010-000-030-096": 1,
    "020-000-033-111": 2,
    "020-000-032-221": 3
  }

  def __init__(self, seq_len=32):
    all_x = []
    all_y = []
    series_x = []
    series_y = []
    all_feats = []
    all_labels = []
    with open("data/person/ConfLongDemo_JSI.txt", "r") as f:
      current_person = "A01"
      for line in f:
        arr = line.split(",")
        if (len(arr) < 6):
          break
        if (arr[0] != current_person):
          # Enque and reset
          series_x = np.stack(series_x, axis=0)
          series_y = np.array(series_y, dtype=np.int32)
          all_x.append(series_x)
          all_y.append(series_y)
          series_x = []
          series_y = []
        current_person = arr[0]
        sensor_id = self.sensor_ids[arr[1]]
        label_col = self.class_map[arr[7].replace("\n", "")]
        feature_col_2 = np.array(arr[4:7], dtype=np.float32)
        feature_col_1 = np.zeros(4, dtype=np.float32)
        feature_col_1[sensor_id] = 1
        feature_col = np.concatenate([feature_col_1, feature_col_2])
        # 100ms sampling time
        # print("feature_col: ",str(feature_col))
        series_x.append(feature_col)
        all_feats.append(feature_col)
        all_labels.append(_one_hot(label_col, 7))
        series_y.append(label_col)

    all_labels = np.stack(all_labels, axis=0)
    print("all_labels.shape: ", str(all_labels.shape))
    prior = np.mean(all_labels, axis=0)
    print("Resampled Prior: ", str(prior * 100))
    all_feats = np.stack(all_feats, axis=0)
    print("all_feats.shape: ", str(all_feats.shape))

    all_mean = np.mean(all_feats, axis=0)
    all_std = np.std(all_feats, axis=0)
    all_mean[3:] = 0
    all_std[3:] = 1
    print("all_mean: ", str(all_mean))
    print("all_std: ", str(all_std))

    inc = seq_len // 2
    sequences_x = []
    sequences_y = []
    for i in range(len(all_x)):
      x, y = all_x[i], all_y[i]
      for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])
    all_x, all_y = np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)

    total_seqs = all_x.shape[1]
    print("Total number of training sequences: {}".format(total_seqs))
    permutation = np.random.RandomState(27731).permutation(total_seqs)
    valid_size = int(0.1 * total_seqs)
    test_size = int(0.15 * total_seqs)

    self.valid_x = all_x[:, permutation[:valid_size]]
    self.valid_y = all_y[:, permutation[:valid_size]]
    self.test_x = all_x[:, permutation[valid_size:valid_size + test_size]]
    self.test_y = all_y[:, permutation[valid_size:valid_size + test_size]]
    self.train_x = all_x[:, permutation[valid_size + test_size:]]
    self.train_y = all_y[:, permutation[valid_size + test_size:]]

    print("Total number of test sequences: {}".format(self.test_x.shape[1]))

  def iterate_train(self, batch_size=16):
    total_seqs = self.train_x.shape[1]
    permutation = np.random.permutation(total_seqs)
    total_batches = total_seqs // batch_size

    for i in range(total_batches):
      start = i * batch_size
      end = start + batch_size
      batch_x = self.train_x[:, permutation[start:end]]
      batch_y = self.train_y[:, permutation[start:end]]
      yield (batch_x, batch_y)


class PowerData:
  def __init__(self, seq_len=32):
    all_x = []
    with open("data/power/household_power_consumption.txt", "r") as f:
      lineno = -1
      memory = [i for i in range(7)]
      for line in f:
        lineno += 1
        if (lineno == 0):
          continue
        arr = line.split(";")
        if (len(arr) < 8):
          continue
        feature_col = arr[2:]
        feature_col, memory = _convert_to_floats(feature_col, memory)
        all_x.append(np.array(feature_col, dtype=np.float32))
    all_x = np.stack(all_x, axis=0)
    all_x -= np.mean(all_x, axis=0)  # normalize
    all_x /= np.std(all_x, axis=0)  # normalize
    all_y = all_x[:, 0].reshape([-1, 1])
    all_x = all_x[:, 1:]
    x, y = all_x, all_y

    self.train_x, self.train_y = _cut_in_sequences(x, seq_len, inc=seq_len, y=y, stack_axis=1)

    print("train_x.shape:", str(self.train_x.shape))
    print("train_y.shape:", str(self.train_y.shape))

    total_seqs = self.train_x.shape[1]
    print("Total number of training sequences: {}".format(total_seqs))
    permutation = np.random.RandomState(23489).permutation(total_seqs)
    valid_size = int(0.1 * total_seqs)
    test_size = int(0.15 * total_seqs)

    self.valid_x = self.train_x[:, permutation[:valid_size]]
    self.valid_y = self.train_y[:, permutation[:valid_size]]
    self.test_x = self.train_x[:, permutation[valid_size:valid_size + test_size]]
    self.test_y = self.train_y[:, permutation[valid_size:valid_size + test_size]]
    self.train_x = self.train_x[:, permutation[valid_size + test_size:]]
    self.train_y = self.train_y[:, permutation[valid_size + test_size:]]

  def iterate_train(self, batch_size=16):
    total_seqs = self.train_x.shape[1]
    permutation = np.random.permutation(total_seqs)
    total_batches = total_seqs // batch_size

    for i in range(total_batches):
      start = i * batch_size
      end = start + batch_size
      batch_x = self.train_x[:, permutation[start:end]]
      batch_y = self.train_y[:, permutation[start:end]]
      yield (batch_x, batch_y)


class SMnistData:
  def __init__(self):
    data = bd.vision.MNIST('./data', split='train', download=True)
    train_x = np.asarray(data.data, dtype=np.float_)
    train_y = np.asarray(data.targets, dtype=np.int_)
    data = bd.vision.MNIST('./data', split='test', download=True)
    test_x = np.asarray(data.data, dtype=np.float_)
    test_y = np.asarray(data.targets, dtype=np.int_)
    train_x = train_x.astype(np.float32) / 255.0
    test_x = test_x.astype(np.float32) / 255.0

    train_split = int(0.9 * train_x.shape[0])
    valid_x = train_x[train_split:]
    train_x = train_x[:train_split]
    valid_y = train_y[train_split:]
    train_y = train_y[:train_split]

    train_x = train_x.reshape([-1, 28, 28])
    test_x = test_x.reshape([-1, 28, 28])
    valid_x = valid_x.reshape([-1, 28, 28])

    self.valid_x = np.transpose(valid_x, (1, 0, 2))
    self.train_x = np.transpose(train_x, (1, 0, 2))
    self.test_x = np.transpose(test_x, (1, 0, 2))
    self.valid_y = valid_y
    self.train_y = train_y
    self.test_y = test_y

    print("Total number of training sequences: {}".format(train_x.shape[0]))
    print("Total number of validation sequences: {}".format(self.valid_x.shape[0]))
    print("Total number of test sequences: {}".format(self.test_x.shape[0]))

  def iterate_train(self, batch_size=16):
    total_seqs = self.train_x.shape[1]
    permutation = np.random.permutation(total_seqs)
    total_batches = total_seqs // batch_size

    for i in range(total_batches):
      start = i * batch_size
      end = start + batch_size
      batch_x = self.train_x[:, permutation[start:end]]
      batch_y = self.train_y[permutation[start:end]]
      yield (batch_x, batch_y)


class TrafficData:
  def __init__(self, seq_len=32):
    df = pd.read_csv("data/traffic/Metro_Interstate_Traffic_Volume.csv")
    holiday = (df["holiday"].values == None).astype(np.float32)
    temp = df["temp"].values.astype(np.float32)
    temp -= np.mean(temp)  # normalize temp by annual mean
    rain = df["rain_1h"].values.astype(np.float32)
    snow = df["snow_1h"].values.astype(np.float32)
    clouds = df["clouds_all"].values.astype(np.float32)
    date_time = df["date_time"].values
    # 2012-10-02 13:00:00
    date_time = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in date_time]
    weekday = np.array([d.weekday() for d in date_time]).astype(np.float32)
    noon = np.array([d.hour for d in date_time]).astype(np.float32)
    noon = np.sin(noon * np.pi / 24)
    features = np.stack([holiday, temp, rain, snow, clouds, weekday, noon], axis=-1)
    traffic_volume = df["traffic_volume"].values.astype(np.float32)
    traffic_volume -= np.mean(traffic_volume)  # normalize
    traffic_volume /= np.std(traffic_volume)  # normalize
    x, y = features, traffic_volume

    train_x, train_y = _cut_in_sequences(x, seq_len, inc=4, y=y)

    self.train_x = np.stack(train_x, axis=0)
    self.train_y = np.stack(train_y, axis=0)
    total_seqs = self.train_x.shape[1]
    print("Total number of training sequences: {}".format(total_seqs))
    permutation = np.random.RandomState(23489).permutation(total_seqs)
    valid_size = int(0.1 * total_seqs)
    test_size = int(0.15 * total_seqs)

    self.valid_x = self.train_x[:, permutation[:valid_size]]
    self.valid_y = self.train_y[:, permutation[:valid_size]]
    self.test_x = self.train_x[:, permutation[valid_size: valid_size + test_size]]
    self.test_y = self.train_y[:, permutation[valid_size: valid_size + test_size]]
    self.train_x = self.train_x[:, permutation[valid_size + test_size:]]
    self.train_y = self.train_y[:, permutation[valid_size + test_size:]]

  def iterate_train(self, batch_size=16):
    total_seqs = self.train_x.shape[1]
    permutation = np.random.permutation(total_seqs)
    total_batches = total_seqs // batch_size

    for i in range(total_batches):
      start = i * batch_size
      end = start + batch_size
      batch_x = self.train_x[:, permutation[start:end]]
      batch_y = self.train_y[:, permutation[start:end]]
      yield (batch_x, batch_y)


