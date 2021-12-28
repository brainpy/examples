# -*- coding: utf-8 -*-

import brainpy as bp


class SimpleNet(bp.Network):
  def __init__(self, syn_model):
    self.pre = bp.models.LIF(1)
    self.post = bp.models.LIF(1)
    self.syn = syn_model(self.pre, self.post, conn=bp.conn.One2One())
    super(SimpleNet, self).__init__(self.syn, self.pre, self.post)


def try_syn_model(model):
  pre = bp.models.LIF(1, V_rest=-60., V_reset=-60., V_th=-40.)
  post = bp.models.LIF(1, V_rest=-60., V_reset=-60., V_th=-40.)
  syn = model(pre, post, conn=bp.conn.One2One())
  net = bp.Network(pre=pre, post=post, syn=syn)

  runner = bp.StructRunner(net,
                           monitors=['pre.V', 'post.V', 'syn.g'],
                           inputs=['pre.input', 22.])
  runner.run(100.)

  fig, gs = bp.visualize.get_figure(1, 2, 4, 8)
  fig.add_subplot(gs[0, 0])
  bp.visualize.line_plot(runner.mon.ts, runner.mon['syn.g'], legend='syn.g')
  fig.add_subplot(gs[0, 1])
  bp.visualize.line_plot(runner.mon.ts, runner.mon['pre.V'], legend='pre.V')
  bp.visualize.line_plot(runner.mon.ts, runner.mon['post.V'], legend='post.V', show=True)


try_syn_model(bp.models.ExpCOBA)
