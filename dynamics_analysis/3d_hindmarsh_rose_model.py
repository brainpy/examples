# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: brainpy
#     language: python
#     name: brainpy
# ---

# %% [markdown]
# # Hindmarsh Rose Model

# %%
import brainpy as bp

bp.math.enable_x64()
bp.math.set_platform('cpu')

# %%
import matplotlib.pyplot as plt
import numpy as np


# %%
class HindmarshRose(bp.DynamicalSystem):
  def __init__(self, method='exp_auto'):
    super(HindmarshRose, self).__init__()

    # parameters
    self.a = 1.
    self.b = 2.5
    self.c = 1.
    self.d = 5.
    self.s = 4.
    self.x_r = -1.6
    self.r = 0.001

    # variables
    self.x = bp.math.Variable(bp.math.ones(1))
    self.y = bp.math.Variable(bp.math.ones(1))
    self.z = bp.math.Variable(bp.math.ones(1))
    self.I = bp.math.Variable(bp.math.zeros(1))

    # integral functions
    def dx(x, t, y, z, Isyn):
      return y - self.a * x ** 3 + self.b * x * x - z + Isyn

    def dy(y, t, x):
      return self.c - self.d * x * x - y

    def dz(z, t, x):
      return self.r * (self.s * (x - self.x_r) - z)

    self.int_x = bp.odeint(f=dx, method=method)
    self.int_y = bp.odeint(f=dy, method=method)
    self.int_z = bp.odeint(f=dz, method=method)

  def update(self, _t, _dt):
    self.x.value = self.int_x(self.x, _t, self.y, self.z, self.I, _dt)
    self.y.value = self.int_y(self.y, _t, self.x, _dt)
    self.z.value = self.int_z(self.z, _t, self.x, _dt)
    self.I[:] = 0.


# %% [markdown]
# ## Simulation

# %%
model = HindmarshRose()

runner = bp.StructRunner(model, monitors=['x', 'y', 'z'], inputs=['I', 1.5])
runner.run(2000.)
bp.visualize.line_plot(runner.mon.ts, runner.mon.x, legend='x', show=True)

# %% [markdown]
# ## Bifurcation analysis

# %%
analyzer = bp.analysis.FastSlow2D(
    [model.int_x, model.int_y, model.int_z],
    fast_vars={'x': [-3, 2], 'y': [-20., 3.]},
    slow_vars={'z': [-0.5, 3.]},
    pars_update={'Isyn': 1.5},
    resolutions={'z': 0.01},
    # options={bp.analysis.C.y_by_x_in_fy: lambda x: model.c - model.d * x * x}
)
analyzer.plot_bifurcation(num_rank=20)
analyzer.plot_trajectory({'x': [1.], 'y': [1.], 'z': [1.]},
                       duration=1700,
                       plot_durations=[360, 1680])
analyzer.show_figure()

# %% [markdown]
# ## Phase plane analysis

# %%
for z in np.arange(0., 2.5, 0.3):
    analyzer = bp.analysis.PhasePlane2D(
      [model.int_x, model.int_y],
      target_vars={'x': [-3, 2], 'y': [-20., 3.]},
      pars_update={'Isyn': 1.5, 'z': z},
      resolutions={'x': 0.01, 'y': 0.01},
    )
    analyzer.plot_nullcline()
    analyzer.plot_vector_field()
    fps = analyzer.plot_fixed_point(with_return=True)
    analyzer.plot_trajectory({'x': [fps[-1, 0] + 0.1], 'y': [fps[-1, 0] + 0.1]},
                             duration=500, plot_durations=[400, 500])
    plt.title(f'z={z:.2f}')
    plt.show()
