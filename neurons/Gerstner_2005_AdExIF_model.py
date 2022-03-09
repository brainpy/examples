# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: brainpy
#     language: python
#     name: brainpy
# ---

# %% [markdown]
# # *(Gerstner, 2005)*: Adaptive Exponential Integrate-and-Fire model

# %% [markdown]
# Adaptive Exponential Integrate-and-Fire neuron model is a spiking model, describes single neuron behavior and can generate many kinds of firing patterns by tuning parameters.

# %%
import brainpy as bp

bp.math.set_dt(0.01)
bp.math.enable_x64()
bp.math.set_platform('cpu')

# %% [markdown]
# ### Tonic

# %%
group = bp.dyn.AdExIF(size=1, a=0., b=60., R=.5, delta_T=2., tau=20., tau_w=30.,
                      V_reset=-55., V_rest=-70, V_th=-30, V_T=-50)

runner = bp.dyn.DSRunner(group, monitors=['V', 'w'], inputs=('input', 65.))
runner.run(500.)
fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel='V', title='tonic')
fig.add_subplot(gs[1, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon.w, ylabel='w', show=True)

# %% [markdown]
# ### Adapting

# %%
group = bp.dyn.AdExIF(size=1, a=0., b=5., R=.5, tau=20., tau_w=100., delta_T=2.,
                      V_reset=-55., V_rest=-70, V_th=-30, V_T=-50)

runner = bp.dyn.DSRunner(group, monitors=['V', 'w'], inputs=('input', 65.))
runner.run(200.)
fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel='V', title='adapting')
fig.add_subplot(gs[1, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon.w, ylabel='w', show=True)

# %% [markdown]
# ### Init Bursting

# %%
group = bp.dyn.AdExIF(size=1, a=.5, b=7., R=.5, tau=5., tau_w=100., delta_T=2.,
                      V_reset=-51, V_rest=-70, V_th=-30, V_T=-50)

runner = bp.dyn.DSRunner(group, monitors=['V', 'w'], inputs=('input', 65.))
runner.run(300.)
fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel='V', ylim=(-55., -35.), title='init_bursting')
fig.add_subplot(gs[1, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon.w, ylabel='w', show=True)

# %% [markdown]
# ### Bursting

# %%
group = bp.dyn.AdExIF(size=1, a=-0.5, b=7., R=.5, delta_T=2., tau=5, tau_w=100,
                      V_reset=-46, V_rest=-70, V_th=-30, V_T=-50)

runner = bp.dyn.DSRunner(group, monitors=['V', 'w'], inputs=('input', 65.))
runner.run(500.)
fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel='V', ylim=(-60., -35.), title='bursting')
fig.add_subplot(gs[1, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon.w, ylabel='w', show=True)

# %% [markdown]
# ### Transient

# %%
group = bp.dyn.AdExIF(size=1, a=1., b=10., R=.5, tau=10, tau_w=100, delta_T=2.,
                      V_reset=-60, V_rest=-70, V_th=-30, V_T=-50)

runner = bp.dyn.DSRunner(group, monitors=['V', 'w'], inputs=('input', 55.))
runner.run(500.)
fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel='V', ylim=(-60., -35.), title='transient')
fig.add_subplot(gs[1, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon.w, ylabel='w', show=True)

# %% [markdown]
# ### Delayed

# %%
group = bp.dyn.AdExIF(size=1, a=-1., b=10., R=.5, delta_T=2., tau=5., tau_w=100.,
                      V_reset=-60, V_rest=-70, V_th=-30, V_T=-50)

runner = bp.dyn.DSRunner(group, monitors=['V', 'w'], inputs=('input', 20.))
runner.run(500.)
fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel='V', ylim=(-60., -35.), title='delayed')
fig.add_subplot(gs[1, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon.w, ylabel='w', show=True)
