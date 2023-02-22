# -*- coding: utf-8 -*-


import argparse
import functools
import os
import time

import brainpy as bp
import brainpy.math as bm
import jax
import jax.numpy as jnp
import numpy as np
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
from torchtoolbox.transform import Cutout

bm.set_environment(bm.TrainingMode())
conv_init = bp.init.KaimingNormal(mode='fan_out', scale=jnp.sqrt(2))
dense_init = bp.init.Normal(0, 0.01)


@jax.custom_vjp
def replace(spike, rate):
  return rate


def replace_fwd(spike, rate):
  return replace(spike, rate), ()


def replace_bwd(res, g):
  return g, g


replace.defvjp(replace_fwd, replace_bwd)


class ScaledWSConv2d(bp.layers.Conv2d):
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               groups=1,
               b_initializer=bp.init.ZeroInit(),
               gain=True,
               eps=1e-4):
    super(ScaledWSConv2d, self).__init__(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         groups=groups,
                                         w_initializer=conv_init,
                                         b_initializer=b_initializer)
    bp.check.is_subclass(self.mode, bm.TrainingMode)
    if gain:
      self.gain = bm.TrainVar(jnp.ones([1, 1, 1, self.out_channels]))
    else:
      self.gain = None
    self.eps = eps

  def update(self, *args):
    assert self.mask is None
    x = args[0] if len(args) == 1 else args[1]
    self._check_input_dim(x)
    w = self.w.value
    fan_in = np.prod(w.shape[:-1])
    mean = jnp.mean(w, axis=[0, 1, 2], keepdims=True)
    var = jnp.var(w, axis=[0, 1, 2], keepdims=True)
    w = (w - mean) / ((var * fan_in + self.eps) ** 0.5)
    if self.gain is not None:
      w = w * self.gain
    y = jax.lax.conv_general_dilated(lhs=bm.as_jax(x),
                                     rhs=bm.as_jax(w),
                                     window_strides=self.stride,
                                     padding=self.padding,
                                     lhs_dilation=self.lhs_dilation,
                                     rhs_dilation=self.rhs_dilation,
                                     feature_group_count=self.groups,
                                     dimension_numbers=self.dimension_numbers)
    return y if self.b is None else (y + self.b.value)


class ScaledWSLinear(bp.layers.Dense):
  def __init__(self,
               in_features,
               out_features,
               b_initializer=bp.init.ZeroInit(),
               gain=True,
               eps=1e-4):
    super(ScaledWSLinear, self).__init__(num_in=in_features,
                                         num_out=out_features,
                                         W_initializer=dense_init,
                                         b_initializer=b_initializer)
    bp.check.is_subclass(self.mode, bm.TrainingMode)
    if gain:
      self.gain = bm.TrainVar(jnp.ones(1, self.num_out))
    else:
      self.gain = None
    self.eps = eps

  def update(self, s, x):
    fan_in = self.W.shape[0]
    mean = jnp.mean(self.W.value, axis=0, keepdims=True)
    var = jnp.var(self.W.value, axis=0, keepdims=True)
    weight = (self.W.value - mean) / ((var * fan_in + self.eps) ** 0.5)
    if self.gain is not None:
      weight = weight * self.gain
    if self.b is not None:
      return x @ weight + self.b
    else:
      return x @ weight


class Scale(bp.layers.Layer):
  def __init__(self, scale: float):
    super(Scale, self).__init__()
    self.scale = scale

  def update(self, s, x):
    return x * self.scale


class WrappedSNNOp(bp.layers.Layer):
  def __init__(self, op):
    super(WrappedSNNOp, self).__init__()
    self.op = op

  def update(self, s, x):
    if s['require_wrap']:
      spike, rate = jnp.split(x, 2, axis=0)
      out = jax.lax.stop_gradient(self.op(s, spike))
      in_for_grad = replace(spike, rate)
      out_for_grad = self.op(s, in_for_grad)
      output = replace(out_for_grad, out)
      return output
    else:
      return self.op(s, x)


class OnlineSpikingVGG(bp.DynamicalSystem):
  cfg = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]

  def __init__(
      self,
      neuron_model,
      weight_standardization=True,
      num_classes=1000,
      neuron_pars: dict = None,
      light_classifier=True,
      batch_norm=False,
      grad_with_rate: bool = False,
      fc_hw: int = 3,
      c_in: int = 3
  ):
    super(OnlineSpikingVGG, self).__init__()

    if neuron_pars is None:
      neuron_pars = dict()
    self.neuron_pars = neuron_pars
    self.neuron_model = neuron_model
    self.grad_with_rate = grad_with_rate
    self.fc_hw = fc_hw

    neuron_sizes = [(32, 32, 64),
                    (32, 32, 128),
                    (16, 16, 256),
                    (16, 16, 256),
                    (8, 8, 512),
                    (8, 8, 512),
                    (4, 4, 512),
                    (4, 4, 512), ]
    neuron_i = 0
    layers = []
    first_conv = True
    in_channels = c_in
    for v in self.cfg:
      if v == 'M':
        layers.append(bp.layers.AvgPool2d(kernel_size=2, stride=2))
      else:
        if weight_standardization:
          conv2d = ScaledWSConv2d(in_channels, v, kernel_size=3, padding=1, stride=1)
          if first_conv:
            first_conv = False
          elif self.grad_with_rate:
              conv2d = WrappedSNNOp(conv2d)
          layers += [conv2d,
                     self.neuron_model(neuron_sizes[neuron_i], **self.neuron_pars),
                     Scale(2.74)]
        else:
          conv2d = bp.layers.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=1, w_initializer=conv_init, )
          if first_conv:
            first_conv = False
          elif self.grad_with_rate:
            conv2d = WrappedSNNOp(conv2d)
          if batch_norm:
            layers += [conv2d,
                       bp.layers.BatchNorm2d(v, momentum=0.9),
                       self.neuron_model(neuron_sizes[neuron_i], **self.neuron_pars)]
          else:
            layers += [conv2d,
                       self.neuron_model(neuron_sizes[neuron_i], **self.neuron_pars),
                       Scale(2.74)]
        neuron_i += 1
        in_channels = v
    self.features = bp.Sequential(*layers)

    if light_classifier:
      self.avgpool = bp.layers.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw))
      if self.grad_with_rate:
        self.classifier = WrappedSNNOp(bp.layers.Dense(512 * self.fc_hw * self.fc_hw,
                                                       num_classes,
                                                       W_initializer=dense_init))
      else:
        self.classifier = bp.layers.Dense(512 * self.fc_hw * self.fc_hw,
                                          num_classes,
                                          W_initializer=dense_init)
    else:
      self.avgpool = bp.layers.AdaptiveAvgPool2d((7, 7))
      if self.grad_with_rate:
        self.classifier = bp.Sequential(
          WrappedSNNOp(ScaledWSLinear(512 * 7 * 7, 4096)),
          neuron_model((4096,), **self.neuron_pars, neuron_dropout=0.0),
          Scale(2.74),
          bp.layers.Dropout(0.5),
          WrappedSNNOp(ScaledWSLinear(4096, 4096)),
          neuron_model((4096,), **self.neuron_pars, neuron_dropout=0.0),
          Scale(2.74),
          bp.layers.Dropout(0.5),
          WrappedSNNOp(bp.layers.Dense(4096, num_classes, W_initializer=dense_init)),
        )
      else:
        self.classifier = bp.Sequential(
          ScaledWSLinear(512 * 7 * 7, 4096),
          neuron_model((4096,), **self.neuron_pars, neuron_dropout=0.0),
          Scale(2.74),
          bp.layers.Dropout(0.5),
          ScaledWSLinear(4096, 4096),
          neuron_model((4096,), **self.neuron_pars, neuron_dropout=0.0),
          Scale(2.74),
          bp.layers.Dropout(0.5),
          bp.layers.Dense(4096, num_classes, W_initializer=dense_init),
        )

  def update(self, s, x):
    if self.grad_with_rate and s['fit']:
      s['require_wrap'] = True
      s['output_type'] = 'spike_rate'
      x = self.features(s, x)
      x = self.avgpool(s, x)
      x = bm.flatten(x, 1)
      x = self.classifier(s, x)
    else:
      s['require_wrap'] = False
      s['output_type'] = 'spike'
      x = self.features(s, x)
      x = self.avgpool(s, x)
      x = bm.flatten(x, 1)
      x = self.classifier(s, x)
    return x


class OnlineIFNode(bp.DynamicalSystem):
  def __init__(
      self,
      size,
      v_threshold: float = 1.,
      v_reset: float = None,
      f_surrogate=bm.surrogate.sigmoid,
      detach_reset: bool = True,
      track_rate: bool = True,
      neuron_dropout: float = 0.0,
      name: str = None,
      mode: bm.Mode = None
  ):
    super().__init__(name=name, mode=mode)
    bp.check.is_subclass(self.mode, bm.TrainingMode)

    self.size = bp.check.is_sequence(size, elem_type=int)
    self.f_surrogate = bp.check.is_callable(f_surrogate)
    self.detach_reset = detach_reset
    self.v_reset = v_reset
    self.v_threshold = v_threshold
    self.track_rate = track_rate
    self.dropout = neuron_dropout

    if self.dropout > 0.0:
      self.rng = bm.random.default_rng()
    self.reset_state(1)

  def reset_state(self, batch_size=1):
    self.v = bp.init.variable_(bm.zeros, self.size, batch_size)
    self.spike = bp.init.variable_(bm.zeros, self.size, batch_size)
    if self.track_rate:
      self.rate_tracking = bp.init.variable_(bm.zeros, self.size, batch_size)

  def update(self, s, x):
    # neuron charge
    self.v.value = jax.lax.stop_gradient(self.v.value) + x
    # neuron fire
    spike = self.f_surrogate(self.v.value - self.v_threshold)
    # spike reset
    spike_d = jax.lax.stop_gradient(spike) if self.detach_reset else spike
    if self.v_reset is None:
      self.v -= spike_d * self.v_threshold
    else:
      self.v.value = (1. - spike_d) * self.v + spike_d * self.v_reset
    # dropout
    if self.dropout > 0.0 and s['fit']:
      mask = self.rng.bernoulli(1 - self.dropout, self.v.shape) / (1 - self.dropout)
      spike = mask * spike
    self.spike.value = spike
    # spike track
    if self.track_rate:
      self.rate_tracking += jax.lax.stop_gradient(spike)
    # output
    if s['output_type'] == 'spike_rate':
      assert self.track_rate
      return jnp.concatenate([spike, self.rate_tracking.value])
    else:
      return spike


class OnlineLIFNode(bp.DynamicalSystem):
  def __init__(
      self,
      size,
      tau: float = 2.,
      decay_input: bool = False,
      v_threshold: float = 1.,
      v_reset: float = None,
      f_surrogate=bm.surrogate.sigmoid,
      detach_reset: bool = True,
      track_rate: bool = True,
      neuron_dropout: float = 0.0,
      name: str = None,
      mode: bm.Mode = None
  ):
    super().__init__(name=name, mode=mode)
    bp.check.is_subclass(self.mode, bm.TrainingMode)

    self.size = bp.check.is_sequence(size, elem_type=int)
    self.tau = tau
    self.decay_input = decay_input
    self.v_threshold = v_threshold
    self.v_reset = v_reset
    self.f_surrogate = f_surrogate
    self.detach_reset = detach_reset
    self.track_rate = track_rate
    self.dropout = neuron_dropout

    if self.dropout > 0.0:
      self.rng = bm.random.default_rng()
    self.reset_state(1)

  def reset_state(self, batch_size=1):
    self.v = bp.init.variable_(bm.zeros, self.size, batch_size)
    self.spike = bp.init.variable_(bm.zeros, self.size, batch_size)
    if self.track_rate:
      self.rate_tracking = bp.init.variable_(bm.zeros, self.size, batch_size)

  def update(self, s, x):
    # neuron charge
    if self.decay_input:
      x = x / self.tau
    if self.v_reset is None or self.v_reset == 0:
      self.v = jax.lax.stop_gradient(self.v.value) * (1 - 1. / self.tau) + x
    else:
      self.v = jax.lax.stop_gradient(self.v.value) * (1 - 1. / self.tau) + self.v_reset / self.tau + x
    # neuron fire
    spike = self.f_surrogate(self.v - self.v_threshold)
    # neuron reset
    spike_d = jax.lax.stop_gradient(spike) if self.detach_reset else spike
    if self.v_reset is None:
      self.v -= spike_d * self.v_threshold
    else:
      self.v = (1. - spike_d) * self.v + spike_d * self.v_reset
    # dropout
    if self.dropout > 0.0 and s['fit']:
      mask = self.rng.bernoulli(1 - self.dropout, spike.shape) / (1 - self.dropout)
      spike = mask * spike
    self.spike.value = spike
    # spike
    if self.track_rate:
      self.rate_tracking.value = jax.lax.stop_gradient(self.rate_tracking * (1 - 1. / self.tau) + spike)
    if s['output_type'] == 'spike_rate':
      assert self.track_rate
      return jnp.concatenate((spike, self.rate_tracking.value))
    else:
      return spike


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


@functools.partial(jax.jit, static_argnums=2)
def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  _, pred = jax.vmap(jax.lax.top_k, in_axes=(0, None))(output, maxk)
  pred = pred.T
  correct = (pred == target.reshape(1, -1)).astype(bm.float_)
  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).sum(0)
    res.append(correct_k * 100.0 / target.size)
  return res


def classify_cifar():
  parser = argparse.ArgumentParser(description='Classify CIFAR')
  parser.add_argument('-T', default=6, type=int, help='simulating time-steps')
  parser.add_argument('-tau', default=2., type=float)
  parser.add_argument('-b', default=128, type=int, help='batch size')
  parser.add_argument('-epochs', default=300, type=int, help='number of total epochs to run')
  parser.add_argument('-j', default=4, type=int, help='number of data loading workers (default: 4)')
  parser.add_argument('-data_dir', type=str, default=r'D:/data')
  parser.add_argument('-dataset', default='cifar10', type=str)
  parser.add_argument('-out_dir', default='./logs', type=str, help='root dir for saving logs and checkpoint')
  parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
  parser.add_argument('-opt', type=str, help='use which optimizer. SGD or Adam', default='SGD')
  parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
  parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
  parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
  parser.add_argument('-step_size', default=100, type=float, help='step_size for StepLR')
  parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
  parser.add_argument('-T_max', default=300, type=int, help='T_max for CosineAnnealingLR')
  parser.add_argument('-drop_rate', type=float, default=0.0)
  parser.add_argument('-weight_decay', type=float, default=0.0)
  parser.add_argument('-loss_lambda', type=float, default=0.05)
  parser.add_argument('-online_update', action='store_true')
  parser.add_argument('-gpu-id', default='0', type=str, help='gpu id')
  args = parser.parse_args()
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

  # datasets
  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    Cutout(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  if args.dataset == 'cifar10':
    dataloader = datasets.CIFAR10
    num_classes = 10
  else:
    dataloader = datasets.CIFAR100
    num_classes = 100
  trainset = dataloader(root=args.data_dir, train=True, download=True, transform=transform_train)
  train_data_loader = data.DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=args.j)
  testset = dataloader(root=args.data_dir, train=False, download=False, transform=transform_test)
  test_data_loader = data.DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=args.j)

  # network
  net = OnlineSpikingVGG(neuron_model=OnlineLIFNode,
                         neuron_pars=dict(tau=args.tau,
                                          neuron_dropout=args.drop_rate,
                                          f_surrogate=bm.surrogate.sigmoid,
                                          track_rate=True,
                                          v_reset=None),
                         weight_standardization=True,
                         num_classes=num_classes,
                         grad_with_rate=True,
                         fc_hw=1,
                         c_in=3)
  print('Total Parameters: %.2fM' % (
      sum(p.size for p in net.vars().subset(bm.TrainVar).unique().values()) / 1000000.0))

  # path
  out_dir = os.path.join(args.out_dir, f'{args.dataset}_T_{args.T}_{args.opt}_lr_{args.lr}_')
  if args.lr_scheduler == 'CosALR':
    out_dir += f'CosALR_{args.T_max}'
  elif args.lr_scheduler == 'StepLR':
    out_dir += f'StepLR_{args.step_size}_{args.gamma}'
  else:
    raise NotImplementedError(args.lr_scheduler)
  if args.online_update:
    out_dir += '_online'
  os.makedirs(out_dir, exist_ok=True)
  with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
    args_txt.write(str(args))

  t_step = args.T

  @bm.to_object(child_objs=net)
  def single_step(x, y, fit=True):
    out = net({'fit': fit}, x)
    if args.loss_lambda > 0.0:
      y = bm.one_hot(y, 10, dtype=bm.float_)
      l = bp.losses.mean_squared_error(out, y) * args.loss_lambda
      l += (1 - args.loss_lambda) * bp.losses.cross_entropy_loss(out, y)
      l /= t_step
    else:
      l = bp.losses.cross_entropy_loss(out, y) / t_step
    return l, out

  @bm.jit
  @bm.to_object(child_objs=net)
  def inference_fun(x, y):
    l, out = bm.for_loop(lambda _: single_step(x, y, False),
                         jnp.arange(t_step),
                         child_objs=net)
    out = out.sum(0)
    n = jnp.sum(jnp.argmax(out, axis=1) == y)
    return l.sum(), n, out

  grad_fun = bm.grad(single_step, grad_vars=net.train_vars().unique(), return_value=True, has_aux=True)

  if args.lr_scheduler == 'StepLR':
    lr = bp.optim.StepLR(args.lr, step_size=args.step_size, gamma=args.gamma)
  elif args.lr_scheduler == 'CosALR':
    lr = bp.optim.CosineAnnealingLR(args.lr, T_max=args.T_max)
  else:
    raise NotImplementedError(args.lr_scheduler)

  if args.opt == 'SGD':
    optimizer = bp.optim.Momentum(lr, net.train_vars().unique(), momentum=args.momentum, weight_decay=args.weight_decay)
  elif args.opt == 'Adam':
    optimizer = bp.optim.AdamW(lr, net.train_vars().unique(), weight_decay=args.weight_decay)
  else:
    raise NotImplementedError(args.opt)

  @bm.jit
  @bm.to_object(child_objs=(optimizer, grad_fun))
  def train_fun(x, y):
    if args.online_update:
      final_loss, final_out = 0., 0.
      for _ in range(t_step):
        grads, l, out = grad_fun(x, y)
        optimizer.update(grads)
        final_loss += l
        final_out += out
    else:
      final_grads, final_loss, final_out = grad_fun(x, y)
      for _ in range(t_step - 1):
        grads, l, out = grad_fun(x, y)
        final_grads = jax.tree_util.tree_map(lambda a, b: a + b, final_grads, grads)
        final_loss += l
        final_out += out
      optimizer.update(final_grads)
    n = jnp.sum(jnp.argmax(final_out, axis=1) == y)
    return final_loss, n, final_out

  start_epoch = 0
  max_test_acc = 0
  if args.resume:
    checkpoint = bp.checkpoints.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    max_test_acc = checkpoint['max_test_acc']

  train_samples = len(train_data_loader)
  test_samples = len(test_data_loader)
  for epoch in range(start_epoch, args.epochs):
    start_time = time.time()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    train_loss = 0
    train_acc = 0
    pbar = tqdm.tqdm(total=train_samples)
    for frame, label in train_data_loader:
      frame = jnp.asarray(frame).transpose(0, 2, 3, 1)
      label = jnp.asarray(label)
      net.reset_state(frame.shape[0])
      batch_loss, n, total_fr = train_fun(frame, label)
      prec1, prec5 = accuracy(total_fr, label, topk=(1, 5))
      train_loss += batch_loss * label.size
      train_acc += n
      losses.update(batch_loss, frame.shape[0])
      top1.update(prec1.item(), frame.shape[0])
      top5.update(prec5.item(), frame.shape[0])

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      # plot progress
      pbar.update(1)
      pbar.set_description(
        'Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
          bt=batch_time.avg, loss=losses.avg, top1=top1.avg, top5=top5.avg,
        )
      )
    pbar.close()

    train_loss /= train_samples
    train_acc /= train_samples
    optimizer.lr.step_epoch()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    test_loss = 0
    test_acc = 0
    pbar = tqdm.tqdm(total=test_samples)
    for frame, label in test_data_loader:
      frame = jnp.asarray(frame).transpose(0, 2, 3, 1)
      label = jnp.asarray(label)
      net.reset_state(frame.shape[0])
      total_loss, n, out = inference_fun(frame, label)
      test_loss += total_loss * label.size
      test_acc += n
      prec1, prec5 = accuracy(out, label, topk=(1, 5))
      losses.update(total_loss, frame.shape[0])
      top1.update(prec1.item(), frame.shape[0])
      top5.update(prec5.item(), frame.shape[0])
      batch_time.update(time.time() - end)
      end = time.time()

      # plot progress
      pbar.update(1)
      pbar.set_description(
        'Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
          bt=batch_time.avg, loss=losses.avg, top1=top1.avg, top5=top5.avg,
        )
      )
    pbar.close()

    test_loss /= test_samples
    test_acc /= test_samples

    if test_acc > max_test_acc:
      max_test_acc = test_acc
      checkpoint = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'max_test_acc': max_test_acc
      }
      bp.checkpoints.save(out_dir, checkpoint, max_test_acc)

    total_time = time.time() - start_time
    print(f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, '
          f'test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, '
          f'total_time={total_time}')


if __name__ == '__main__':
  classify_cifar()
