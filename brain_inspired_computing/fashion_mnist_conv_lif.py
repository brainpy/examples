# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time

import brainpy as bp
import brainpy.math as bm
import brainpy_datasets as bd
import jax
import jax.numpy as jnp
from jax import lax

bm.set_environment(mode=bm.training_mode, dt=1.)


class ConvLIF(bp.DynamicalSystem):
  def __init__(self, n_time: int, n_channel: int, tau: float = 5.):
    super().__init__()
    self.n_time = n_time

    lif_par = dict(keep_size=True, V_rest=0., V_reset=0., V_th=1.,
                   tau=tau, spk_fun=bm.surrogate.arctan)

    self.block1 = bp.Sequential(
      bp.layers.Conv2d(1, n_channel, kernel_size=3, padding=(1, 1), b_initializer=None),
      bp.layers.BatchNorm2d(n_channel, momentum=0.9),
      bp.dyn.Lif((28, 28, n_channel), **lif_par)
    )
    self.block2 = bp.Sequential(
      bp.layers.MaxPool2d(2, 2),  # 14 * 14
      bp.layers.Conv2d(n_channel, n_channel, kernel_size=3, padding=(1, 1), b_initializer=None),
      bp.layers.BatchNorm2d(n_channel, momentum=0.9),
      bp.dyn.Lif((14, 14, n_channel), **lif_par),
    )
    self.block3 = bp.Sequential(
      bp.layers.MaxPool2d(2, 2),  # 7 * 7
      bp.layers.Flatten(),
      bp.layers.Dense(n_channel * 7 * 7, n_channel * 4 * 4, b_initializer=None),
      bp.dyn.Lif(4 * 4 * n_channel, **lif_par),
    )
    self.block4 = bp.Sequential(
      bp.layers.Dense(n_channel * 4 * 4, 10, b_initializer=None),
      bp.dyn.Lif(10, **lif_par),
    )

  def update(self, x):
    # x.shape = [B, H, W, C]
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    return self.block4(x)


class IFNode(bp.DynamicalSystem):
  """The Integrate-and-Fire neuron. The voltage of the IF neuron will
  not decay as that of the LIF neuron. The sub-threshold neural dynamics
  of it is as followed:

  .. math::
      V[t] = V[t-1] + X[t]
  """

  def __init__(self, size: tuple, v_threshold: float = 1., v_reset: float = 0.,
               spike_fun=bm.surrogate.arctan, mode=None, reset_mode='soft'):
    super().__init__(mode=mode)
    bp.check.is_subclass(self.mode, bm.TrainingMode)

    self.size = bp.check.is_sequence(size, elem_type=int, allow_none=False)
    self.reset_mode = bp.check.is_string(reset_mode, candidates=['hard', 'soft'])
    self.v_threshold = bp.check.is_float(v_threshold)
    self.v_reset = bp.check.is_float(v_reset)
    self.spike_fun = bp.check.is_callable(spike_fun)

    # variables
    self.V = bm.Variable(jnp.zeros((1,) + size, dtype=bm.float_), batch_axis=0)

  def reset_state(self, batch_size):
    self.V.value = jnp.zeros((batch_size,) + self.size, dtype=bm.float_)

  def update(self, x):
    self.V.value += x
    spike = self.spike_fun(self.V - self.v_threshold)
    if self.reset_mode == 'hard':
      one = lax.convert_element_type(1., bm.float_)
      self.V.value = self.v_reset * spike + (one - spike) * self.V
    else:
      self.V -= spike * self.v_threshold
    return spike


class ConvIF(bp.DynamicalSystem):
  def __init__(self, n_time: int, n_channel: int):
    super().__init__()
    self.n_time = n_time

    self.block1 = bp.Sequential(
      bp.layers.Conv2d(1, n_channel, kernel_size=3, padding=(1, 1), ),
      bp.layers.BatchNorm2d(n_channel, momentum=0.9),
      IFNode((28, 28, n_channel), spike_fun=bm.surrogate.arctan)
    )
    self.block2 = bp.Sequential(
      bp.layers.MaxPool([2, 2], 2, channel_axis=-1),  # 14 * 14
      bp.layers.Conv2d(n_channel, n_channel, kernel_size=3, padding=(1, 1), ),
      bp.layers.BatchNorm2d(n_channel, momentum=0.9),
      IFNode((14, 14, n_channel), spike_fun=bm.surrogate.arctan),
    )
    self.block3 = bp.Sequential(
      bp.layers.MaxPool([2, 2], 2, channel_axis=-1),  # 7 * 7
      bp.layers.Flatten(),
      bp.layers.Dense(n_channel * 7 * 7, n_channel * 4 * 4, ),
      IFNode((4 * 4 * n_channel,), spike_fun=bm.surrogate.arctan),
    )
    self.block4 = bp.Sequential(
      bp.layers.Dense(n_channel * 4 * 4, 10, ),
      IFNode((10,), spike_fun=bm.surrogate.arctan),
    )

  def update(self, x):
    x = self.block1(x)  # x.shape = [B, H, W, C]
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    return x


class TrainMNIST:
  def __init__(self, net, n_time):
    self.net = net
    self.n_time = n_time
    self.f_opt = bp.optim.Adam(bp.optim.ExponentialDecay(0.2, 1, 0.9999),
                               train_vars=net.train_vars().unique())
    self.f_grad = bm.grad(self.loss, grad_vars=self.f_opt.vars_to_train,
                          has_aux=True, return_value=True)

  def inference(self, X, fit=False):
    def run_net(t):
      bp.share.save(t=t, fit=fit)
      return self.net(X)

    self.net.reset_state(X.shape[0])
    return bm.for_loop(run_net, jnp.arange(self.n_time, dtype=bm.float_), jit=False)

  def loss(self, X, Y, fit=False):
    fr = bm.max(self.inference(X, fit), axis=0)
    ys_onehot = bm.one_hot(Y, 10, dtype=bm.float_)
    l = bp.losses.mean_squared_error(fr, ys_onehot)
    n = bm.sum(fr.argmax(1) == Y)
    return l, n

  @bm.cls_jit
  def f_predict(self, X, Y):
    return self.loss(X, Y, fit=False)

  @bm.cls_jit
  def f_train(self, X, Y):
    bp.share.save(fit=True)
    grads, l, n = self.f_grad(X, Y)
    self.f_opt.update(grads)
    return l, n


def main():
  parser = argparse.ArgumentParser(description='Classify Fashion-MNIST')
  parser.add_argument('-platform', default='cpu', help='platform')
  parser.add_argument('-model', default='lif', help='Neuron model to use')
  parser.add_argument('-n_time', default=4, type=int, help='simulating time-steps')
  parser.add_argument('-tau', default=5., type=float, help='LIF time constant')
  parser.add_argument('-batch', default=128, type=int, help='batch size')
  parser.add_argument('-n_channel', default=128, type=int, help='channels of ConvLIF')
  parser.add_argument('-n_epoch', default=64, type=int, metavar='N', help='number of total epochs to run')
  parser.add_argument('-data-dir', default='d:/data', type=str, help='root dir of Fashion-MNIST dataset')
  parser.add_argument('-out-dir', default='./logs', type=str, help='root dir for saving logs and checkpoint')
  parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
  args = parser.parse_args()
  print(args)

  bm.set_platform(args.platform)

  # net
  if args.model == 'if':
    net = ConvIF(n_time=args.n_time, n_channel=args.n_channel)
    out_dir = os.path.join(args.out_dir,
                           f'{args.model}_T{args.n_time}_b{args.batch}'
                           f'_lr{args.lr}_c{args.n_channel}')
  elif args.model == 'lif':
    net = ConvLIF(n_time=args.n_time, n_channel=args.n_channel, tau=args.tau)
    out_dir = os.path.join(args.out_dir,
                           f'{args.model}_T{args.n_time}_b{args.batch}'
                           f'_lr{args.lr}_c{args.n_channel}_tau{args.tau}')
  else:
    raise ValueError

  trainer = TrainMNIST(net, args.n_time)

  # dataset
  train_set = bd.vision.FashionMNIST(root=args.data_dir, split='train', download=True)
  test_set = bd.vision.FashionMNIST(root=args.data_dir, split='test', download=True)
  x_train = jnp.asarray(train_set.data / 255, dtype=bm.float_).reshape((-1, 28, 28, 1))
  y_train = jnp.asarray(train_set.targets, dtype=bm.int_)
  x_test = jnp.asarray(test_set.data / 255, dtype=bm.float_).reshape((-1, 28, 28, 1))
  y_test = jnp.asarray(test_set.targets, dtype=bm.int_)

  os.makedirs(out_dir, exist_ok=True)
  with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
    args_txt.write(str(args))
    args_txt.write('\n')
    args_txt.write(' '.join(sys.argv))

  max_test_acc = -1
  for epoch_i in range(0, args.n_epoch):
    start_time = time.time()
    loss, train_acc = [], 0.
    for i in range(0, x_train.shape[0], args.batch):
      xs = x_train[i: i + args.batch]
      ys = y_train[i: i + args.batch]
      with jax.disable_jit():
        l, n = trainer.f_train(xs, ys)
      loss.append(l)
      train_acc += n
    train_acc /= x_train.shape[0]
    train_loss = jnp.mean(jnp.asarray(loss))
    trainer.f_opt.lr.step_epoch()

    loss, test_acc = [], 0.
    for i in range(0, x_test.shape[0], args.batch):
      xs = x_test[i: i + args.batch]
      ys = y_test[i: i + args.batch]
      l, n = trainer.f_predict(xs, ys)
      loss.append(l)
      test_acc += n
    test_acc /= x_test.shape[0]
    test_loss = jnp.mean(jnp.asarray(loss))

    t = (time.time() - start_time) / 60
    print(f'epoch {epoch_i}, used {t:.3f} min, '
          f'train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f}, '
          f'test_loss = {test_loss:.4f}, test_acc = {test_acc:.4f}')

    if max_test_acc < test_acc:
      max_test_acc = test_acc
      states = {
        'net': net.state_dict(),
        'optimizer': trainer.f_opt.state_dict(),
        'epoch_i': epoch_i,
        'train_acc': train_acc,
        'test_acc': test_acc,
      }
      bp.checkpoints.save_pytree(os.path.join(out_dir, 'fmnist-conv-lif.bp'), states)

  # inference
  state_dict = bp.checkpoints.load_pytree(os.path.join(out_dir, 'fmnist-conv-lif.bp'))
  net.load_state_dict(state_dict['net'])
  correct_num = 0
  for i in range(0, x_test.shape[0], 512):
    xs = x_test[i: i + 512]
    ys = y_test[i: i + 512]
    correct_num += trainer.f_predict(xs, ys)[1]
  print('Max test accuracy: ', correct_num / x_test.shape[0])


if __name__ == '__main__':
  main()
