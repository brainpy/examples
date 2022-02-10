# -*- coding: utf-8 -*-
# # Mackey-Glass equation

# The Mackey-Glass equation is the nonlinear time delay differential equation
#
# $$
# \frac{dx}{dt} = \beta \frac{ x_{\tau} }{1+{x_{\tau}}^n}-\gamma x, \quad \gamma,\beta,n > 0,
# $$
#
# where $\beta, \gamma, \tau, n$ are real numbers, and $x_{\tau}$ represents the value of the variable $x$ at time $(t−\tau)$. Depending on the values of the parameters, this equation displays a range of periodic and chaotic dynamics. 

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

assert bp.__version__ >= '2.0.2'

bm.set_dt(0.05)


class MackeyGlassEq(bp.NeuGroup):
  def __init__(self, num, beta=2., gamma=1., tau=2., n=9.65):
    super(MackeyGlassEq, self).__init__(num)

    # parameters
    self.beta = beta
    self.gamma = gamma
    self.tau = tau
    self.n = n

    # variables
    self.x = bp.ConstantDelay(num, delay=tau)
    self.x.data[:] = 1.2 + 0.2 * (bm.random.random(num) - 0.5)
    self.x_latest = bm.Variable(self.x.latest)
    self.x_oldest = bm.Variable(self.x.oldest)

    # functions
    self.integral = bp.odeint(lambda x, t, x_tau: self.beta * x_tau / (1 + x_tau ** self.n) - self.gamma * x,
                              method='exp_auto')

  def update(self, _t, _dt):
    self.x_oldest.value = self.x.pull()
    self.x_latest.value = self.integral(self.x_latest, _t, self.x_oldest, _dt)
    self.x.push(self.x_latest)
    self.x.update(_t, _dt)


eq = MackeyGlassEq(1, beta=0.2, gamma=0.1, tau=17, n=10)
# eq = MackeyGlassEq(1, )

runner = bp.StructRunner(eq, monitors=['x_latest', 'x_oldest'])
runner.run(1000)

plt.plot(runner.mon.x_latest[:, 0], runner.mon.x_oldest[:, 0])
plt.show()

