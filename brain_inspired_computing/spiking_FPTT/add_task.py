import argparse
import math
import time

import brainpy as bp
import brainpy.math as bm
import numba
import numpy as np


@numba.njit(fastmath=True, nogil=True)
def _adding_problem_generator(X_num, X_mask, Y, N, seq_len=6, number_of_ones=2):
  for i in numba.prange(N):
    positions1 = np.random.choice(np.arange(math.floor(seq_len / 2)),
                                  size=math.floor(number_of_ones / 2),
                                  replace=False)
    positions2 = np.random.choice(np.arange(math.ceil(seq_len / 2), seq_len),
                                  size=math.ceil(number_of_ones / 2),
                                  replace=False)
    for p in positions1:
      X_mask[p, i] = 1
      Y[i, 0] += X_num[p, i, 0]
    for p in positions2:
      X_mask[p, i] = 1
      Y[i, 0] += X_num[p, i, 0]


class AddTask:
  def __init__(self, seq_len=6, high=1, number_of_ones=2):
    self.seq_len = seq_len
    self.high = high
    self.number_of_ones = number_of_ones

  def __call__(self, batch_size):
    x_num = np.random.uniform(low=0, high=self.high, size=(self.seq_len, batch_size, 1))
    x_mask = np.zeros((self.seq_len, batch_size, 1))
    ys = np.ones((batch_size, 1))
    _adding_problem_generator(x_num, x_mask, ys, batch_size, self.seq_len, self.number_of_ones)
    xs = np.append(x_num, x_mask, axis=2)
    return xs, ys


class IF(bp.DynamicalSystemNS):
  def __init__(
      self, size, V_th=0.5,
      spike_fun: bm.surrogate.Surrogate = bm.surrogate.MultiGaussianGrad()
  ):
    super().__init__()

    self.size = size
    self.V_th = V_th
    self.spike_fun = spike_fun
    self.reset_state(self.mode)

  def reset_state(self, batch_size=1):
    self.V = bp.init.variable_(bm.zeros, self.size, batch_size)
    self.spike = bp.init.variable_(bm.zeros, self.size, batch_size)

  def update(self, x, tau_m):
    mem = self.V + (-self.V + x) * tau_m
    self.spike.value = self.spike_fun(mem - self.V_th)
    self.V.value = (1 - self.spike) * mem


class SNN(bp.DynamicalSystemNS):
  def __init__(self, input_size, hidden_size):
    super().__init__()

    self.lin_inp = bp.layers.Linear(input_size + hidden_size, hidden_size)
    self.lin_tau = bp.layers.Linear(hidden_size + hidden_size, hidden_size)
    self.rnn = IF(hidden_size)
    self.out = bp.layers.Linear(hidden_size, 1, W_initializer=bp.init.XavierNormal())
    self.act = bp.layers.Sigmoid()

    self.loss_func = bp.losses.MSELoss()

  def _step(self, x):
    inp = self.lin_inp(bm.cat((x, self.rnn.spike), dim=-1))
    tau = self.act(self.lin_tau(bm.cat((inp, self.rnn.V), dim=-1)))
    self.rnn(inp, tau)

  def update(self, xs, y):
    # xs: (num_time, num_batch, num_hidden)
    bm.for_loop(self._step, xs)
    out = self.out(self.rnn.V)
    out = out.squeeze()
    y = y.squeeze()
    loss = self.loss_func(out, y)
    return loss, out


class FPTT_Trainer:
  def __init__(
      self,
      net: bp.DynamicalSystemNS,
      opt: bp.optim.Optimizer,
      clip: float,
      alpha: float = 0.1,
      beta: float = 0.5,
      rho: float = 0.0,
  ):
    super().__init__()
    self.alpha = alpha
    self.beta = beta
    self.rho = rho
    self.clip = clip

    # objects
    self.net = net
    self.opt = opt
    opt.register_train_vars(net.train_vars().unique())

    # parameters
    self.named_params = {}
    for name, param in self.opt.vars_to_train.items():
      sm = bm.Variable(param.clone())
      lm = bm.Variable(bm.zeros_like(param))
      self.named_params[name] = (sm, lm)

  def reset_params(self):
    for name, param in self.opt.vars_to_train.items():
      param.value = self.named_params[name][0].value

  def update_params(self):
    for name, param in self.opt.vars_to_train.items():
      sm, lm = self.named_params[name]
      lm += (-self.alpha * (param - sm))
      sm *= (1.0 - self.beta)
      sm += (self.beta * param - (self.beta / self.alpha) * lm)

  def dyn_loss(self, lambd=1.):
    regularization = 0.
    for name, param in self.opt.vars_to_train.items():
      sm, lm = self.named_params[name]
      regularization += (self.rho - 1.) * bm.sum(param * lm)
      regularization += lambd * 0.5 * self.alpha * bm.sum(bm.square(param - sm))
    return regularization

  def f_loss(self, xs, ys, progress):
    l, _ = self.net(xs, ys)
    reg = self.dyn_loss()
    return l * progress + reg, (l, reg)

  @bm.cls_jit
  def predict(self, xs, ys):
    return self.net(xs, ys)[0]

  @bm.cls_jit
  def fit(self, xs, ys, progress):
    grads, (loss, reg) = bm.grad(self.f_loss, grad_vars=self.opt.vars_to_train, has_aux=True)(xs, ys, progress)
    grads = bm.clip_by_norm(grads, self.clip)
    self.opt.update(grads)
    self.update_params()
    return loss, reg


class BPTT_Trainer:
  def __init__(
      self,
      net: bp.DynamicalSystemNS,
      opt: bp.optim.Optimizer,
      clip: float,
  ):
    super().__init__()
    self.clip = clip

    # objects
    self.net = net
    self.opt = opt
    opt.register_train_vars(net.train_vars().unique())

  def f_loss(self, xs, ys):
    l, _ = self.net(xs, ys)
    return l

  @bm.cls_jit
  def predict(self, xs, ys):
    return self.net(xs, ys)[0]

  @bm.cls_jit
  def fit(self, xs, ys):
    grads, loss = bm.grad(self.f_loss, grad_vars=self.opt.vars_to_train, return_value=True)(xs, ys)
    grads = bm.clip_by_norm(grads, self.clip)
    self.opt.update(grads)
    return loss


def fptt_training():
  parser = argparse.ArgumentParser()
  parser.add_argument('--alpha', type=float, default=.1, help='Alpha')
  parser.add_argument('--beta', type=float, default=0.5, help='Beta')
  parser.add_argument('--rho', type=float, default=0.0, help='Rho')
  parser.add_argument('--bptt', type=int, default=300, help='sequence length')
  parser.add_argument('--nhid', type=int, default=128, help='number of hidden units per layer')
  parser.add_argument('--lr', type=float, default=3e-3, help='initial learning rate (default: 4e-3)')
  parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
  parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit (default: 200)')
  parser.add_argument('--parts', type=int, default=10, help='Parts to split the sequential input into (default: 10)')
  parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
  parser.add_argument('--save', type=str, default='', help='path of model to save')
  parser.add_argument('--load', type=str, default='', help='path of model to load')
  parser.add_argument('--wdecay', type=float, default=0.0, help='weight decay')
  parser.add_argument('--seed', type=int, default=1111, help='random seed')
  parser.add_argument('--optim', type=str, default='adam', help='optimizer to use')
  args = parser.parse_args()

  bm.random.seed(args.seed)

  # model
  with bm.environment(mode=bm.TrainingMode(batch_size=args.batch_size)):
    model = SNN(2, args.nhid)

  # dataset
  dataset = AddTask(args.bptt, number_of_ones=2)

  # optimizer
  if args.optim == 'adam':
    optimizer = bp.optim.Adam(lr=args.lr, weight_decay=args.wdecay)
  elif args.optim == 'sgd':
    optimizer = bp.optim.SGD(lr=args.lr, weight_decay=args.wdecay)
  else:
    raise ValueError

  # trainer
  trainer = FPTT_Trainer(model, optimizer, args.clip, alpha=args.alpha, beta=args.beta, rho=args.rho)

  # loading
  if args.load:
    states = bp.checkpoints.load_pytree(args.load)
    model.load_state_dict(states['model'])
    optimizer.load_state_dict(states['opt'])

  # training
  step = args.bptt // args.parts
  for epoch in range(1, args.epochs + 1):
    model.reset_state(args.batch_size)

    # fitting
    s_t = time.time()
    x, y = dataset(args.batch_size)
    losses, regs = [], []
    for p in range(0, args.parts):
      start = p * step
      l, r = trainer.fit(x[start: start + step], y, (p + 1) / args.parts)
      losses.append(l.item())
      regs.append(r.item())

    # prediction
    x, y = dataset(args.batch_size)
    loss_act = trainer.predict(x, y).item()
    print(f'Epoch {epoch}, '
          f'time {time.time() - s_t:.4f} s, '
          f'train loss {np.mean(np.array(losses)):.4f}, '
          f'train reg {np.mean(np.array(regs)):.4f}, '
          f'test loss {loss_act:.4f}. ')
    trainer.reset_params()
    trainer.opt.lr.step_epoch()


def bptt_training():
  parser = argparse.ArgumentParser()
  parser.add_argument('--bptt', type=int, default=300, help='sequence length')
  parser.add_argument('--nhid', type=int, default=128, help='number of hidden units per layer')
  parser.add_argument('--lr', type=float, default=3e-3, help='initial learning rate (default: 4e-3)')
  parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
  parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit (default: 200)')
  parser.add_argument('--parts', type=int, default=10, help='Parts to split the sequential input into (default: 10)')
  parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
  parser.add_argument('--save', type=str, default='', help='path of model to save')
  parser.add_argument('--load', type=str, default='', help='path of model to load')
  parser.add_argument('--wdecay', type=float, default=0.0, help='weight decay')
  parser.add_argument('--seed', type=int, default=1111, help='random seed')
  parser.add_argument('--optim', type=str, default='adam', help='optimizer to use')
  args = parser.parse_args()

  bm.random.seed(args.seed)

  # model
  with bm.environment(mode=bm.TrainingMode(batch_size=args.batch_size)):
    model = SNN(2, args.nhid)

  # dataset
  dataset = AddTask(args.bptt, number_of_ones=2)

  # optimizer
  if args.optim == 'adam':
    optimizer = bp.optim.Adam(lr=args.lr, weight_decay=args.wdecay)
  elif args.optim == 'sgd':
    optimizer = bp.optim.SGD(lr=args.lr, weight_decay=args.wdecay)
  else:
    raise ValueError

  # trainer
  trainer = BPTT_Trainer(model, optimizer, args.clip)

  # loading
  if args.load:
    states = bp.checkpoints.load_pytree(args.load)
    model.load_state_dict(states['model'])
    optimizer.load_state_dict(states['opt'])

  # training
  for epoch in range(1, args.epochs + 1):
    model.reset_state(args.batch_size)

    # fitting
    s_t = time.time()
    x, y = dataset(args.batch_size)
    loss = trainer.fit(x, y).item()

    # prediction
    x, y = dataset(args.batch_size)
    loss_act = trainer.predict(x, y).item()

    print(f'Epoch {epoch}, '
          f'time {time.time() - s_t:.4f} s, '
          f'train loss {loss:.4f}, '
          f'test loss {loss_act:.4f}. ')
    trainer.opt.lr.step_epoch()


if __name__ == '__main__':
    # fptt_training()
    bptt_training()

