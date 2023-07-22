import argparse
import time
import numpy as np

import brainpy as bp
import brainpy.math as bm
import brainpy_datasets as bdata
from functools import partial


class SigmoidBeta(bp.DynamicalSystemNS):
  def __init__(self, alpha=1., is_train=False):
    super(SigmoidBeta, self).__init__()
    if alpha is None:
      self.alpha = bm.asarray(1.)  # create a tensor out of alpha
    else:
      self.alpha = bm.asarray(alpha)  # create a tensor out of alpha
    if is_train:
      self.alpha = bm.TrainVar(self.alpha)

  def update(self, x):
    return bm.sigmoid(self.alpha * x)


class LTC_LIF(bp.DynamicalSystemNS):
  def __init__(
      self, input_size, hidden_size, beta=1.8, b0=0.1,
      spike_fun: bm.surrogate.Surrogate = bm.surrogate.MultiGaussianGrad()
  ):
    super().__init__()

    self.hidden_size = hidden_size
    self.beta = beta
    self.b0 = b0
    self.spike_fun = spike_fun

    self.lin = bp.layers.Linear(input_size, hidden_size, W_initializer=bp.init.XavierNormal())
    self.lr = bp.layers.Linear(hidden_size, hidden_size, W_initializer=bp.init.Orthogonal())
    self.tauM = bp.layers.Linear(hidden_size + hidden_size, hidden_size,
                                 W_initializer=bp.init.XavierNormal())
    self.tauAdp = bp.layers.Linear(hidden_size + hidden_size, hidden_size,
                                   W_initializer=bp.init.XavierNormal())
    self.actM = SigmoidBeta(is_train=True)
    self.actAdp = SigmoidBeta(is_train=True)

    self.reset_state(self.mode)

  def reset_state(self, batch_size=1):
    self.V = bp.init.variable_(bm.zeros, self.hidden_size, batch_size)
    self.b = bp.init.variable_(bm.zeros, self.hidden_size, batch_size)
    self.spike = bp.init.variable_(bm.zeros, self.hidden_size, batch_size)

  def update(self, x):
    encoding = self.lin(x) + self.lr(self.spike)
    tauM = self.actM(self.tauM(bm.cat((encoding, self.V), dim=-1)))
    tauAdp = self.actAdp(self.tauAdp(bm.cat((encoding, self.b), dim=-1)))
    b = tauAdp * self.b + (1 - tauAdp) * self.spike
    self.b.value = b
    B = self.b0 + self.beta * b
    d_mem = -self.V + encoding
    mem = self.V + d_mem * tauM
    spike = self.spike_fun(mem - B)
    self.V.value = (1 - spike) * mem
    self.spike.value = spike
    return spike


class ReadoutLIF(bp.DynamicalSystemNS):
  def __init__(
      self, input_size, hidden_size, beta=1.8, b0=0.1,
  ):
    super().__init__()

    self.size = hidden_size
    self.beta = beta
    self.b0 = b0

    self.lin = bp.layers.Linear(input_size, hidden_size,
                                W_initializer=bp.init.XavierNormal())
    self.tauM = bp.layers.Linear(hidden_size + hidden_size, hidden_size,
                                 W_initializer=bp.init.XavierNormal())
    self.actM = SigmoidBeta(is_train=True)

    self.reset_state(self.mode)

  def reset_state(self, batch_size=1):
    self.V = bp.init.variable_(bm.zeros, self.size, batch_size)

  def update(self, x):
    encoding = self.lin(x)
    tauM = self.actM(self.tauM(bm.cat((encoding, self.V), dim=-1)))
    mem = (1 - tauM) * self.V + tauM * encoding
    self.V.value = mem
    return mem


class SNN(bp.DynamicalSystemNS):
  def __init__(self, input_size, hidden_size, output_size):
    super(SNN, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.layer1 = LTC_LIF(input_size, hidden_size)
    self.layer2 = LTC_LIF(hidden_size, hidden_size)
    self.layer3 = ReadoutLIF(hidden_size, output_size)

    self.fr = bm.Variable(bm.asarray(0.))

  def update(self, x):
    x = bm.expand_dims(x, axis=1)
    spk_1 = self.layer1(x)
    spk_2 = self.layer2(spk_1)
    out = self.layer3(spk_2)
    self.fr += (spk_1.mean() + spk_2.mean())
    return out


class FPTT_Trainer:
  def __init__(
      self,
      net: bp.DynamicalSystem,
      optimizer: bp.optim.Optimizer,

      debias: bool = False,
      clip: float = 0.,
      alpha: float = 0.1,
      beta: float = 0.5,
      rho: float = 0.,
  ):
    self.clip = clip
    self.alpha = alpha
    self.beta = beta
    self.rho = rho
    self.debias = debias

    self.optimizer = optimizer
    self.net = net
    optimizer.register_train_vars(net.train_vars().unique())

    # parameters
    self.named_params = {}
    for name, param in optimizer.vars_to_train.items():
      sm = bm.Variable(param.clone())
      lm = bm.Variable(bm.zeros_like(param))
      if debias:
        dm = bm.Variable(bm.zeros_like(param))
        self.named_params[name] = (sm, lm, dm)
      else:
        self.named_params[name] = (sm, lm)

  def reset_params(self):
    if not self.debias:
      for name, param in self.optimizer.vars_to_train.items():
        param.value = self.named_params[name][0].value

  def update_params(self, epoch):
    for name, param in self.optimizer.vars_to_train.items():
      if self.debias:
        sm, lm, dm = self.named_params[name]
        beta = (1. / (1. + epoch))
        sm *= (1.0 - beta)
        sm += (beta * param)
        dm *= (1. - beta)
        dm += (beta * lm)
      else:
        sm, lm = self.named_params[name]
        lm += (-self.alpha * (param - sm))
        sm *= (1.0 - self.beta)
        sm += (self.beta * param - (self.beta / self.alpha) * lm)

  def dyn_loss(self, lambd=1.):
    reg = 0.
    for name, param in self.optimizer.vars_to_train.items():
      if self.debias:
        sm, lm, dm = self.named_params[name]
        reg += (self.rho - 1.) * bm.sum(param * lm)
        reg += (1. - self.rho) * bm.sum(param * dm)
      else:
        sm, lm = self.named_params[name]
        reg += (self.rho - 1.) * bm.sum(param * lm)
        reg += lambd * 0.5 * self.alpha * bm.sum(bm.square(param - sm))
    return reg

  def _loss(self, x, y, progress):
    out = self.net(x)
    loss = progress * bp.losses.cross_entropy_loss(out, y, reduction='mean')
    reg = self.dyn_loss()
    return loss + reg, (loss, reg)

  def _train(self, x, progress, y, epoch):
    grads, (loss, reg) = bm.grad(self._loss, grad_vars=self.optimizer.vars_to_train, has_aux=True)(
      x, y, progress
    )
    if self.clip > 0.:
      grads = bm.clip_by_norm(grads, self.clip)
    self.optimizer.update(grads)
    self.update_params(epoch)
    return loss, reg

  @bm.cls_jit
  def fit(self, xs, ys, epoch):  # xs: (num_time, num_batch)
    progresses = bm.linspace(0., 1., xs.shape[0])
    loss, reg = bm.for_loop(partial(self._train, epoch=epoch, y=ys), (xs, progresses), remat=True)
    return loss, reg


parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0.1, help='Alpha')
parser.add_argument('--beta', type=float, default=0.5, help='Beta')
parser.add_argument('--rho', type=float, default=0.0, help='Rho')
parser.add_argument('--debias', action='store_true', help='FedDyn debias algorithm')

parser.add_argument('--bptt', type=int, default=300, help='sequence length')
parser.add_argument('--nhid', type=int, default=256, help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=5e-3, help='initial learning rate (default: 4e-3)')
parser.add_argument('--clip', type=float, default=1., help='gradient clipping')

parser.add_argument('--epochs', type=int, default=250, help='upper epoch limit (default: 200)')
parser.add_argument('--batch_size', type=int, default=512, metavar='N', help='batch size')

parser.add_argument('--wdecay', type=float, default=0., help='weight decay')
parser.add_argument('--optim', type=str, default='adam', help='optimizer to use')
parser.add_argument('--when', nargs='+', type=int, default=[10, 30, 50, 75, 90], help='When to decay the learning rate')
parser.add_argument('--load', type=str, default='', help='path to load the model')
parser.add_argument('--save', type=str, default='./models/', help='path to load the model')
parser.add_argument('--permute', action='store_true', help='use permuted dataset (default: False)')
args = parser.parse_args()

bm.set(mode=bm.TrainingMode(args.batch_size))  # important

# datasets
n_classes = 10
train_data = bdata.vision.MNIST(r'/mnt/d/data/',  download=True, split='train')
x_train = (train_data.data / 255.).reshape(-1, 28 * 28)
y_train = train_data.targets


def train_data():
  indices = np.random.permutation(len(x_train))
  for i in range(0, len(x_train), args.batch_size):
    idx = indices[i: i + args.batch_size]
    yield x_train[idx].T, y_train[idx]


# model
model = SNN(1, args.nhid, n_classes)

# optimizer
lr = bp.optim.MultiStepLR(args.lr, args.when, gamma=0.1)
if args.optim == 'adam':
  optimizer = bp.optim.Adam(lr=lr, weight_decay=args.wdecay)
elif args.optim == 'sgd':
  optimizer = bp.optim.SGD(lr=lr, weight_decay=args.wdecay)
else:
  raise ValueError

# trainer
trainer = FPTT_Trainer(model,
                       optimizer,
                       debias=args.debias,
                       clip=args.clip,
                       alpha=args.alpha,
                       beta=args.beta,
                       rho=args.rho)

# training
for epoch in range(1, args.epochs + 1):
  num_data = 0
  for data, target in train_data():
    t0 = time.time()
    model.reset_state(data.shape[0])
    losses, regs = trainer.fit(data, target, epoch)
    total_reg_loss = regs.sum().item()
    total_clf_loss = losses.sum().item()

    num_data += data.shape[0]
    print(
      f'Epoch {epoch} [{num_data}/{len(x_train)}]\t'
      f'time: {time.time() - t0:.4f}s\tLoss: {total_clf_loss:.6f}\t'
      f'Reg: {total_reg_loss:.6f}'
    )
