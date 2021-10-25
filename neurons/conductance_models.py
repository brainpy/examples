# -*- coding: utf-8 -*-

import brainmodels
import brainpy as bp

trn = bp.CondNeuGroup(ina=brainmodels.Na.INa(g_max=100., E=50., V_sh=-55.),
                      ik=brainmodels.K.IDR(g_max=10., E=-95., V_sh=-55.),
                      ca=brainmodels.Ca.DynCa(tau=5., C_rest=2.4e-4),
                      it=brainmodels.Ca.ICaT_RE(g_max=2.),
                      il=brainmodels.other.IL(g_max=0.05, E=-77.),
                      ikl=brainmodels.other.IKL(g_max=0.005, E=-95.))
trn.init(1, monitors=['ina.p', 'it.p', 'it.q', 'V'])
trn = bp.math.jit(trn)

trn.run(1000, inputs=('input', 1.5))
fig, gs = bp.visualize.get_figure(2, 1, 4, 10)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(trn.mon.ts, trn.mon['V'], legend='V')
fig.add_subplot(gs[1, 0])
bp.visualize.line_plot(trn.mon.ts, trn.mon['ina.p'], legend='ina.p')
bp.visualize.line_plot(trn.mon.ts, trn.mon['it.p'], legend='it.p')
bp.visualize.line_plot(trn.mon.ts, trn.mon['it.p'], legend='it.q', show=True)

